import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from scipy import stats, sparse
from numpy.random import default_rng

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_grad_enabled(False)

def MLP(input_dim, output_dim, hidden_layers_dims, activation, last_layer_activated, bias):
    layers = []
    layers.append(torch.nn.Linear(input_dim, hidden_layers_dims[0], bias=bias))
    layers.append(activation)
    for i in range(1, len(hidden_layers_dims)):
        layers.append(torch.nn.Linear(hidden_layers_dims[i - 1], hidden_layers_dims[i], bias=bias))
        layers.append(activation)
    layers.append(torch.nn.Linear(hidden_layers_dims[-1], output_dim, bias=bias))
    if last_layer_activated:
        layers.append(activation)
    return torch.nn.Sequential(*layers)

def propagate_features(network_state, W, network_thinking_time, recurrent_activation_function, additive_update, persistent_observation=None, feature_transformation_model=None):
    with torch.no_grad():
        network_state = torch.tensor(network_state, dtype=torch.float64)
        persistent_observation = torch.tensor(persistent_observation, dtype=torch.float64) if persistent_observation is not None else None
        W = torch.tensor(W, dtype=torch.float64)

        if recurrent_activation_function is None:
            activation = None
        elif recurrent_activation_function == "tanh":
            activation = np.tanh
        else:
            raise ValueError("Activation function not available.")

        for step in range(network_thinking_time):
            if additive_update:
                network_state += W.T @ network_state
            else:
                network_state = W.T @ network_state

            if feature_transformation_model is not None:
                network_state = feature_transformation_model(network_state)
            elif activation is not None:
                network_state = activation(network_state)

            if persistent_observation is not None:
                network_state[: persistent_observation.shape[0]] = persistent_observation

        return network_state.detach().numpy()

# def query_pairs_of_node_embeddings(W, network_state, self_link_allowed=False):
#     node_embeddings_concatenated_dict = {}
#     idx = np.arange(len(W))

#     W = abs(W)
#     links = np.clip((np.tril(W) + np.triu(W).T), 0, 1)
#     if not self_link_allowed:
#         np.fill_diagonal(links, 0)

#     for i in range(len(W)):
#         nbr = links[i] > 0
#         for j in idx[nbr]:
#             concatenated_features = np.concatenate([network_state[i], network_state[j]])
#             node_embeddings_concatenated_dict[len(node_embeddings_concatenated_dict)] = {
#                 "from_node": i,
#                 "to_node": j,
#                 "concatenated_features": concatenated_features,
#             }

#     return node_embeddings_concatenated_dict, np.array([node_embeddings_concatenated_dict[e]["concatenated_features"] for e in node_embeddings_concatenated_dict])

def predict_new_nodes(growth_decision_model, embeddings_for_growth_model, node_embedding_size):
    new_nodes_predictions = []
    with torch.no_grad():
        predictions_probabilities = growth_decision_model(torch.tensor(embeddings_for_growth_model, dtype=torch.float64)).detach().numpy()
        new_nodes_predictions = (predictions_probabilities > 0).squeeze()

    return new_nodes_predictions

def update_weights(G, network_state, model, config):
    with torch.no_grad():
        for i, j in G.edges():
            G[i][j]["weight"] = model(torch.tensor(np.concatenate([network_state[i], network_state[j]]), dtype=torch.float64)).detach().numpy()[0]
    return G

def add_new_nodes(G, config, network_state, node_embeddings_concatenated_dict, new_nodes_predictions, node_based_growth, node_pairs_based_growth):
    current_graph_size = len(G)
    if node_pairs_based_growth:
        for idx_edge in node_embeddings_concatenated_dict:
            if new_nodes_predictions[idx_edge]:
                target_connections = (
                    node_embeddings_concatenated_dict[idx_edge]["from_node"],
                    node_embeddings_concatenated_dict[idx_edge]["to_node"],
                )

                neighbors = np.unique(np.concatenate([[n for n in nx.all_neighbors(G, target_connections[0])], [n for n in nx.all_neighbors(G, target_connections[1])]]))

                G.add_node(current_graph_size)
                G.add_edge(target_connections[0], current_graph_size, weight=1 if config["binary_connectivity"] else 0)
                G.add_edge(current_graph_size, target_connections[1], weight=1 if config["binary_connectivity"] else 0)

                network_state = np.concatenate([network_state, np.expand_dims(np.mean(network_state[neighbors], axis=0), axis=0)])

                current_graph_size += 1

    elif node_based_growth:
        if len(G) == 1:
            neighbors = np.array([[0]])
        else:
            neighbors = []
            for idx_node in range(len(G)):
                neighbors_idx = [n for n in nx.all_neighbors(G, idx_node)]
                neighbors_idx.append(idx_node)
                neighbors.append(np.unique(neighbors_idx))

        for idx_node in range(len(G)):
            if new_nodes_predictions.shape == ():
                new_nodes_predictions = new_nodes_predictions.reshape(1)

            if new_nodes_predictions[idx_node]:
                if len(neighbors) != 0:
                    G.add_node(current_graph_size)
                    for neighbor in neighbors[idx_node]:
                        if nx.is_directed(G):
                            G.add_edge(neighbor, current_graph_size, weight=1)
                            G.add_edge(current_graph_size, neighbor, weight=1)
                        else:
                            G.add_edge(neighbor, current_graph_size, weight=1)
                    current_graph_size += 1

                    network_state = np.concatenate([network_state, np.expand_dims(np.mean(network_state[neighbors[idx_node]], axis=0), axis=0)])

    return G, network_state

# Load the CSV files
cells_data = pd.read_csv('/home/pakhi/Documents/gsoc/ndp/ndp_for_gnn/birth_death_timings.csv')
positions_data = pd.read_csv('/home/pakhi/Documents/gsoc/ndp/ndp_for_gnn/cell_positions.csv')

# Initialize an empty graph
G = nx.Graph()

# Extract all unique time stamps from the birth column
time_stamps = sorted(cells_data['Birth'].dropna().unique())

# Dictionary to store the state of the graph at each time stamp
graph_snapshots = {}

# Define a distance threshold for creating edges
distance_threshold = 100.0  

# Function to compute Euclidean distance between two cells
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

# NDP function to grow the network based on birth timestamps
def NDP(cells_data, positions_data, G, distance_threshold):
    for time in time_stamps:
        new_cells = cells_data[cells_data['Birth'] == time]
        
        for _, row in new_cells.iterrows():
            parent = row['Parent']
            cell = row['Cell']
            
            if parent in G:
                G.remove_node(parent)
            
            pos_row = positions_data[positions_data['Parent Cell'] == cell]
            if not pos_row.empty:
                pos = (pos_row['parent_x'].values[0], pos_row['parent_y'].values[0], pos_row['parent_z'].values[0])
                G.add_node(cell, pos=pos)
        
        for cell1 in G.nodes(data=True):
            for cell2 in G.nodes(data=True):
                if cell1[0] != cell2[0]:
                    pos1 = np.array(cell1[1]['pos'])
                    pos2 = np.array(cell2[1]['pos'])
                    distance = euclidean_distance(pos1, pos2)
                    if distance < distance_threshold:
                        G.add_edge(cell1[0], cell2[0], weight=distance)
        
        graph_snapshots[time] = G.copy()

# Call NDP function to grow the network
NDP(cells_data, positions_data, G, distance_threshold)

# Function to plot the graph
def plot_graph(graph, time):
    pos = {node: data['pos'][:2] for node, data in graph.nodes(data=True)}
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='black', linewidths=1, font_size=15)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title(f"Cell Development at Time {time}")
    plt.show()

# Plot the graph at different time stamps
for time in time_stamps[:10]:  
    plot_graph(graph_snapshots[time], time)
