import networkx as nx
import numpy as np
import torch
from scipy import stats, sparse
from numpy.random import default_rng


class growing_graph:
    def __init__(self, G: nx.Graph, feat_dim, num_nodes, edges):
        self.G = G
        self.feat_dim = feat_dim
        self.num_nodes = num_nodes
        self.edges = edges
        self.graph = nx.Graph()

    def generate_initial_graph(
        self, network_size, sparsity, binary_connectivity, undirected, seed
    ):
        nb_disjoint_initial_graphs = np.inf
        while nb_disjoint_initial_graphs > 1:
            maxWeight = 1
            minWeight = -1
            rng = default_rng(seed)
            if binary_connectivity:
                rvs = stats.uniform(loc=0, scale=1).rvs
                W = np.rint(
                    sparse.random(
                        network_size,
                        network_size,
                        density=sparsity,
                        data_rvs=rvs,
                        random_state=rng,
                    ).toarray()
                )
            else:
                rvs = stats.uniform(loc=minWeight, scale=maxWeight - minWeight).rvs
                W = sparse.random(
                    network_size,
                    network_size,
                    density=sparsity,
                    data_rvs=rvs,
                    random_state=rng,
                ).toarray()  # rows are outbounds, columns are inbounds

            np.fill_diagonal(W, 0)
            disjoint_initial_graphs = [
                e for e in nx.connected_components(nx.from_numpy_array(W))
            ]
            nb_disjoint_initial_graphs = len(disjoint_initial_graphs)

        if undirected:
            self.G = nx.from_numpy_array(W, create_using=nx.Graph)
        else:
            self.G = nx.from_numpy_array(W, create_using=nx.DiGraph)

        return self.G, nx.adjacency_matrix(self.G).toarray()

    def add_new_nodes(self, config, new_nodes_prediction, current_graph_size):
        if len(self.G) == 1:
            neighbours = np.array([[0]])
        else:
            neighbours = []
            for idx_node in range(len(self.G)):
                neighbours_idx = [n for n in nx.all_neighbors(self.G, idx_node)]
                neighbours_idx.append(idx_node)
                neighbours.append(np.unique(neighbours_idx))

        # continue to add new nodes based on prediciton
        for idx_node in range(len(self.G)):
            if new_nodes_prediction.shape == ():
                new_nodes_prediction = new_nodes_prediction.reshape(1)

            if new_nodes_prediction[idx_node]:
                if len(neighbours) != 0:
                    self.G.add_node(current_graph_size)
                    for neighbour in neighbours[idx_node]:
                        self.G.add_edge(neighbour, len(current_graph_size), weight=1)

                    current_graph_size += 1

                    network_state = np.concatenate(
                        [
                            network_state,
                            np.expand_dims(
                                np.mean(network_state[neighbours[idx_node]], axis=0),
                                axis=0,
                            ),
                        ]
                    )

        return self.G, network_state

    def mlp(
        self,
        input_dim,
        output_dim,
        hidden_layers_dims,
        activation,
        last_layer_activated,
        bias,
    ):
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_layers_dims[0], bias=bias))
        layers.append(activation)

        for i in range(1, len(hidden_layers_dims)):
            layers.append(
                torch.nn.Linear(
                    hidden_layers_dims[i - 1], hidden_layers_dims[i], bias=bias
                )
            )
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

    def pair_embeddings(self, W, network_state, self_link=False):
        node_embeddings_concat_dict = {}
        idx = np.arange(len(W))

        W = abs(W)
        links = np.clip(np.tril(W) + np.triu(W).T, 0, 1)
        if not self_link:
            np.fill_diagonal(links, 0)

        for i in range(len(W)):
            nbr = links[i] > 0

            for j in idx[nbr]:
                concat_feat = np.concatenate([network_state[i], network_state[j]])
                node_embeddings_concat_dict[len(node_embeddings_concat_dict)] = {
                    "from_node": i,
                    "to_node": j,
                    "concat_feat": concat_feat,
                }

        return node_embeddings_concat_dict

    def predict_new_nodes(self, config, model, concat_node_embeddings):
        new_nodes = []
        with torch.no_grad():
            prediction = (
                model(torch.tensor(concat_node_embeddings, dtype=torch.float32))
                .detach()
                .numpy()
            )
            new_nodes = (prediction > 0).squeeze()

        return new_nodes

    def update_weights(self, network_state, config, model):
        with torch.no_grad():
            for i, j in self.G.edges():
                self.G[i][j]["weight"] = (
                    model(
                        torch.tensor(
                            np.concatenate([network_state[i], network_state[j]]),
                            dtype=torch.float32,
                        )
                    )
                    .detach()
                    .numpy()[0]
                )

        return self.G



# new

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


# Load the CSV files
birth_death_df = pd.read_csv('birth_death_timings.csv')
positions_df = pd.read_csv('cell_positions.csv')

# Sort the birth_death_df by birth time to get the sequence of cell divisions
birth_death_df = birth_death_df.sort_values(by='birth_time')

# Create a dictionary to store the coordinates of each cell
cell_positions = {row['Parent Cell']: (row['parent_x'], row['parent_y'], row['parent_z']) for index, row in positions_df.iterrows()}

# Function to compute Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))


# Modified NDP function to grow the network based on birth timestamps
def NDP():
    G = nx.Graph()
    cell_coordinates = {}
    
    for index, row in birth_death_df.iterrows():
        cell_name = row['cell_name']
        parent_cell_name = row['parent_cell_name']
        
        # Add the cell to the graph
        G.add_node(cell_name)
        
        # If the cell has a parent, add an edge based on Euclidean distance
        if parent_cell_name in cell_coordinates:
            parent_coords = cell_coordinates[parent_cell_name]
            cell_coords = cell_positions[cell_name]
            G.add_edge(cell_name, parent_cell_name, weight=euclidean_distance(parent_coords, cell_coords))
        
        # Store the coordinates of the cell
        cell_coordinates[cell_name] = cell_positions[cell_name]
    
    return G


# Function to predict the coordinates of daughter cells based on parent cell's coordinates using MLP
def predict_coordinates(mlp_model, parent_coords):
    parent_coords_tensor = torch.tensor(parent_coords, dtype=torch.float32)
    predicted_coords = mlp_model(parent_coords_tensor)
    return predicted_coords.detach().numpy()


# Example MLP model for predicting cell coordinates
input_dim = 3  # Coordinates are in 3D space (x, y, z)
output_dim = 6  # Two sets of coordinates for the two daughter cells
hidden_layers_dims = [64, 64]
activation = torch.nn.ReLU()
last_layer_activated = False
bias = True

mlp_model = MLP(input_dim, output_dim, hidden_layers_dims, activation, last_layer_activated, bias).to(device)

# Load an example parent cell's coordinates and predict the coordinates of daughter cells
parent_cell_name = 'P0'
parent_coords = cell_positions[parent_cell_name]
predicted_daughter_coords = predict_coordinates(mlp_model, parent_coords)

# Evaluate the predictions
actual_daughter_coords = [cell_positions['P1'], cell_positions['P2']]  # Replace with actual daughter cell names
evaluation_error = np.mean([euclidean_distance(predicted_daughter_coords[:3], actual_daughter_coords[0]),
                            euclidean_distance(predicted_daughter_coords[3:], actual_daughter_coords[1])])

print(f"Prediction error: {evaluation_error}")

# Generate the network based on birth timestamps
G = NDP()

# Visualize the network (optional)
import matplotlib.pyplot as plt
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
