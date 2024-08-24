import random
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


# from growing_nn.graph.directed_graph import DirectedGraph


class GraphNCA(nn.Module):
    def __init__(self, graph, num_hidden_channels: int = 16, max_replications: int = 2):
        super().__init__()
        self.graph = graph
        
        # self.num_input_nodes = self.graph.num_input_nodes
        # self.num_output_nodes = self.graph.num_output_nodes

        # self.input_nodes = self.graph.input_nodes
        # self.output_nodes = self.graph.output_nodes

        self.value_idx = 0
        self.replication_idx = 1

        self.operations = [torch.add, torch.subtract, torch.multiply]
        self.activations = [torch.relu, torch.tanh]

        self.replicated_cells = []
        self.num_operations = len(self.operations)
        self.num_activations = len(self.activations)

        self.operation_channels = [2, 4]
        self.activation_channels = [5, 6]

        self.num_hidden_channels = num_hidden_channels
        self.num_channels = self.get_number_of_channels(
            self.num_operations, self.num_activations, self.num_hidden_channels
        )

        self.perception_net = GCNConv(
            self.num_channels, self.num_channels * 3, bias=False
        )
        self.update_net = nn.Sequential(
            nn.Linear(self.num_channels * 3, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_channels),
        )
        self.split_network = nn.Sequential(        
            nn.Linear(self.num_channels, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_channels * 2),
        )
        self.max_replications = max_replications

    @classmethod
    def get_number_of_channels(
        cls, num_operations: int, num_activations: int, num_hidden_channels
    ):
        return num_operations + num_activations + num_hidden_channels + 2

    def forward(self, x, edge_index):
        features = self.perception_net(x, edge_index)
        update = self.update_net(features)
        x = x + update
        return x
    
    # def to_data(self):
    #     edges = []

    #     for node in self.edge_dict:
    #         destinations = self.edge_dict[node]
    #         for d in destinations:
    #             edges.append([node, d])

    #     edges = torch.tensor(edges).long().t().contiguous().to(self.nodes.device)
        
    #     return Data(
    #         x=self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device),
    #         edge_index=edges,
    #     )

    # def replicate(self, x, edge_dict):
    #     num_nodes = x.shape[0]
    #     current_count = num_nodes

    #     # ready_to_replicate = x[:, self.replication_idx] > replication_threshold
    #     # ready_to_replicate = ready_to_replicate.squeeze().detach().cpu().numpy()
    #     dist = Bernoulli(logits=x[:, self.replication_idx])
    #     ready_to_replicate = dist.sample().squeeze()
    #     ready_to_replicate_indices = [
    #         i
    #         for i in range(len(ready_to_replicate))
    #         if ready_to_replicate[i] == 1.0 and i not in self.output_nodes
    #     ]
    #     ready_to_replicate_indices = ready_to_replicate_indices[: self.max_replications]

    #     if len(ready_to_replicate_indices) > 0:

    #         children = self.replication_network(x[ready_to_replicate_indices])

    #         new_edge_dict = {}
    #         for parent_node in ready_to_replicate_indices:
    #             parent_destinations = edge_dict[parent_node]

    #             random_destination = random.choices(parent_destinations, k=1)
    #             new_edge_dict[
    #                 current_count
    #             ] = random_destination  # set child node to new destination
    #             new_edge_dict[parent_node] = [current_count]  # add edge to parent node
    #             current_count += 1

    #         return children, new_edge_dict
    #     return None, None

    def grow(
        self,
        graph,
        parent_index,
        daughter_labels,
    ):
        new_graph = graph.copy()
        # for i in range(num_iterations):
        #     data = new_graph.to_data()
        #     x = self.forward(data.x, data.edge_index)

        #     if replicate_interval is not None:
        #         if i % replicate_interval == 0:
        #             children, new_edge_dict = self.replicate(
        #                 x,
        #                 new_graph.edge_dict,
        #             )
        #             if children is not None:
        #                 new_graph.add_nodes(children)
        #                 new_graph.add_edges(new_edge_dict)

        data = new_graph.to_data()
        # print("data.x.shape",data.x.shape)

        x = self.forward(data.x, data.edge_index)

        split = self.split_network(x[parent_index])

        #store the split values in two daughter cells variable, daughter 1 and daughter 2
        
        daughter1 = split[:self.num_channels]
        daughter2 = split[self.num_channels:]
        daughters = torch.stack([daughter1, daughter2])

        #add daughter cells to the graph and remove the parent cell from the graph
        new_graph = new_graph.add_daughter_cells(daughters, parent_index, daughter_labels)

        #create edges between the parent cell and the daughter cells and pass old edge disctionary too
        new_graph = new_graph.add_edges(new_graph.edge_dict, parent_index)
        print("grown graph",new_graph.nodes.shape)

        return new_graph
