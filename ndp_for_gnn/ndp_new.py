import numpy as np
import pandas as pd
import torch
import networkx as nx
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


def propagate_features(
    self,
    network_state,
    network_thinking_time,
    activation_function,
    additive_update,
    feature_transformation_model,
    persistent_observation,
):
    with torch.no_grad():
        network_state = torch.tensor(network_state, dtype=torch.float32)
        persistent_observation = (
            torch.tensor(persistent_observation, dtype=torch.float32)
            if persistent_observation is not None
            else None
        )

        W = nx.adjacency_matrix(self.G).toarray()
        W = torch.tensor(W, dtype=torch.float32)
        for step in range(network_thinking_time):
            if additive_update:
                network_state += W.T @ network_state
            else:
                network_state = W.T @ network_state

            if feature_transformation_model is not None:
                network_state = feature_transformation_model(network_state)
            elif activation_function is not None:
                network_state = activation_function(network_state)

            if persistent_observation is not None:
                network_state[
                    0 : persistent_observation.shape[0]
                ] = persistent_observation

    return network_state.detach().numpy()


def query_pairs_of_node_embeddings(W, network_state, self_link_allowed=False):
    node_embeddings_concatenated_dict = {}
    idx = np.arange(len(W))

    W = abs(W)
    links = np.clip((np.tril(W) + np.triu(W).T), 0, 1)
    if not self_link_allowed:
        np.fill_diagonal(links, 0)

    for i in range(len(W)):
        nbr = links[i] > 0
        for j in idx[nbr]:
            concatenated_features = np.concatenate([network_state[i], network_state[j]])
            node_embeddings_concatenated_dict[len(node_embeddings_concatenated_dict)] = {
                "from_node": i,
                "to_node": j,
                "concatenated_features": concatenated_features,
            }

    return node_embeddings_concatenated_dict, np.array([node_embeddings_concatenated_dict[e]["concatenated_features"] for e in node_embeddings_concatenated_dict])

def update_weights(G, network_state, model, config):
    with torch.no_grad():
        for i, j in G.edges():
            G[i][j]["weight"] = model(torch.tensor(np.concatenate([network_state[i], network_state[j]]), dtype=torch.float64)).detach().numpy()[0]
    return G



def generate_initial_graph(cells_data, positions_data, initial_size, distance_threshold):
    """
    This function generates an initial graph based on the data from CSV files.

    Args:
        cells_data (pd.DataFrame): DataFrame containing birth and death timings of cells.
        positions_data (pd.DataFrame): DataFrame containing positions of cells.
        initial_size (int): The initial number of nodes in the network.
        distance_threshold (float): The distance threshold for creating edges.

    Returns:
        G: networkx.Graph: The initial graph.
        W: np.array: The initial adjacency matrix.
    """
    # Extract the initial set of cells to form the graph
    initial_cells = cells_data.head(initial_size)

    # Initialize an empty graph
    G = nx.Graph()

    for _, row in initial_cells.iterrows():
        cell = row['Cell']
        pos_row = positions_data[positions_data['Parent Cell'] == cell]
        if not pos_row.empty:
            pos = (pos_row['parent_x'], pos_row['parent_y'], pos_row['parent_z'])
            G.add_node(cell, pos=pos)

    # Create edges based on distance threshold
    for cell1 in G.nodes(data=True):
        for cell2 in G.nodes(data=True):
            if cell1[0] != cell2[0]:
                pos1 = np.array(cell1[1]['pos'])
                pos2 = np.array(cell2[1]['pos'])
                distance = euclidean_distance(pos1, pos2)
                if distance < distance_threshold:
                    G.add_edge(cell1[0], cell2[0], weight=distance)

    # Convert the graph to an adjacency matrix
    W = nx.adjacency_matrix(G).toarray()

    return G, W



# Load the CSV files
cells_data = pd.read_csv('/home/pakhi/Documents/gsoc/ndp/ndp_for_gnn/birth_death_timings.csv')
positions_data =  pd.read_csv('/home/pakhi/Documents/gsoc/ndp/ndp_for_gnn/cell_positions.csv')


cell_positions = {row['Parent Cell']: (row['parent_x'], row['parent_y'], row['parent_z']) for index, row in positions_data.iterrows()}

# Sort the birth_death_df by birth time to get the sequence of cell divisions
time_stamps = sorted(cells_data['Birth'].dropna().unique())

# Dictionary to store the state of the graph at each time stamp
graph_snapshots = {}

# Define a distance threshold for creating edges
distance_threshold = 100.0  

# Function to compute Euclidean distance between two cells
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

# NDP function to grow the network based on birth timestamps
def add_new_nodes():
    G = nx.Graph()

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

    return G



# Function to predict the coordinates of daughter cells based on parent cell's coordinates using MLP
def predict_coordinates(mlp_model, parent_coords):
    parent_coords_tensor = torch.tensor(parent_coords, dtype=torch.float32)
    predicted_coords = mlp_model(parent_coords_tensor)
    return predicted_coords.detach().numpy()


# input_dim = 3 
# output_dim = 6  
# hidden_layers_dims = [64, 64]
# activation = torch.nn.ReLU()
# last_layer_activated = False
# bias = True

# mlp_model = MLP(input_dim, output_dim, hidden_layers_dims, activation, last_layer_activated, bias).to(device)

# Load an example parent cell's coordinates and predict the coordinates of daughter cells
# parent_cell_name = 'P0'
# parent_coords = cell_positions[parent_cell_name]
# predicted_daughter_coords = predict_coordinates(mlp_model, parent_coords)

# Evaluate the predictions
# actual_daughter_coords = [cell_positions['P1'], cell_positions['P2']]  # Replace with actual daughter cell names
# evaluation_error = np.mean([euclidean_distance(predicted_daughter_coords[:3], actual_daughter_coords[0]),
#                             euclidean_distance(predicted_daughter_coords[3:], actual_daughter_coords[1])])

# print(f"Prediction error: {evaluation_error}")

# Generate the network based on birth timestamps
# G = NDP()

# Visualize the network (optional)
# import matplotlib.pyplot as plt
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True)
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
