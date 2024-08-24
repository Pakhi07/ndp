import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch_geometric.data import Data

# from growing_nn.graph.generated_network import GeneratedNetwork


class Graph:
    def __init__(self, nodes, edge_dict, labels):
        self.nodes = nodes
        self.edge_dict = edge_dict
        self.labels = labels
        
    def add_edges(self, edge_dict, parent_index):
        #add edges for new daughter cells and connect the daughter cells to the nodes that parent cell was connected to and create new edge dictionary
        for node in edge_dict:
            if parent_index in edge_dict[node]:
                edge_dict[node].remove(parent_index)
                edge_dict[node].append(self.nodes.size(0) - 2)
                edge_dict[node].append(self.nodes.size(0) - 1)
        edge_dict[self.nodes.size(0) - 2] = [parent_index]
        edge_dict[self.nodes.size(0) - 1] = [parent_index]
        self.edge_dict = edge_dict
        return Graph(self.nodes, self.edge_dict, self.labels)

        

    def add_daughter_cells(self, daughters, parent_index, daughter_labels):
        print(parent_index)
        self.labels[parent_index] = daughter_labels[0]
        self.labels.update({self.nodes.size(0) : daughter_labels[1]})
        print("labels",self.labels)
        #add daughter cells to the graph and remove the parent cell from the graph
        return self.add_nodes(daughters, parent_index)
        # self.remove_node(parent_index)
        
    # def remove_node(self, node_index):
    #     #remove a node from the graph
    #     del self.edge_dict[node_index]
    #     for node in self.edge_dict:
    #         if node_index in self.edge_dict[node]:
    #             self.edge_dict[node].remove(node_index)

    def add_nodes(self, new_nodes, parent_index):
        if new_nodes.dim() == 1:
            new_nodes = new_nodes.unsqueeze(0) 
        # self.nodes = torch.cat([self.nodes, new_nodes])
        # split the array at the parent index into two arrays
        left_nodes = self.nodes[:parent_index]
        right_nodes = self.nodes[parent_index:]
        #remove the parent node from the right array
        right_nodes = right_nodes[1:]
        # concatenate the two arrays with the new nodes in between
        self.nodes = torch.cat([left_nodes, new_nodes, right_nodes])
        print(self.nodes)
        return Graph(self.nodes, self.edge_dict, self.labels)

       


    def to_data(self):
        edges = []
        for node in self.edge_dict:
            destinations = self.edge_dict[node]
            for d in destinations:
                edges.append([node, d])

        edges = torch.tensor(edges).long().t().contiguous().to(self.nodes.device)
        return Data(
            x=self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device),
            edge_index=edges,
        )

    # A recursive function used by topologicalSort
    # def topologicalSortUtil(self, v, visited, stack):

    #     # Mark the current node as visited.
    #     visited[v] = True

    #     # Recur for all the vertices adjacent to this vertex
    #     if v in self.edge_dict:
    #         for i in self.edge_dict[v]:
    #             if not visited[i]:
    #                 self.topologicalSortUtil(i, visited, stack)

    #     # Push current vertex to stack which stores result
    #     stack.append(v)

    # # The function to do Topological Sort. It uses recursive
    # # topologicalSortUtil()
    # def topological_sort(self):
    #     # Mark all the vertices as not visited
    #     self.V = self.nodes.size(0)
    #     visited = [False] * self.V
    #     stack = []

    #     # Call the recursive helper function to store Topological
    #     # Sort starting from all vertices one by one
    #     for i in range(self.V):
    #         if not visited[i]:
    #             self.topologicalSortUtil(i, visited, stack)

    #     # Print contents of the stack
    #     return stack[::-1]

    def plot(self, fig=None, node_colors=None):
        data = self.to_data()
        G = torch_geometric.utils.to_networkx(data, to_undirected=True)

        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")

        if fig is None:
            fig = plt.figure()
        canvas = FigureCanvas(fig)

        # if node_colors is None:
        #     node_colors = ["blue"] * self.nodes.size(0)
        #     for i in self.input_nodes:
        #         node_colors[i] = "green"
        #     for i in self.output_nodes:
        #         node_colors[i] = "red"

        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, labels=self.labels)

        canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    # def generate_network(self, *args, **kwargs):
    #     return GeneratedNetwork(self, *args, **kwargs)

    def copy(self):
        nodes = self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device)
        edge_dict = copy.deepcopy(self.edge_dict)
        labels = copy.deepcopy(self.labels)
        return Graph(
            nodes, edge_dict, labels
        )
