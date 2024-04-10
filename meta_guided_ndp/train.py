import copy

from NDP import growing_graph as meta_ndp
from utils import seed_python_numpy_torch_cuda, mnist_data_loader, image_to_patch
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import torch.nn as nn


def mnist_eval(G: nx.Graph, config: dict, seed: int = None):
    render = False
    policy_connectivity = nx.to_numpy_array(G)
    mnist_loader = mnist_data_loader()
    criterion = nn.CrossEntropyLoss()
    eval_loss = 0
    try:
        diameter = nx.diameter(G.to_undirected())
    except:
        diameter = int(np.sqrt(len(G)))
        print(
            f"WARNING: Graph is not connected due to prunning. Diameter manually set to {diameter}."
        )
    policy_connectivity = nx.to_numpy_array(G)

    network_thinking_time = diameter + config["network_thinking_time_extra_rollout"]

    for image, label in tqdm(mnist_loader):
        image = image_to_patch(image=image, patch_size=config["patch_size"])
        network_state[: config["observation_dim"]] = image
        persistent_observation = (
            image if config["persistent_observation_rollout"] else None
        )
        network_state = meta_ndp.propagate_features(
            network_state=network_state,
            W=policy_connectivity,
            network_thinking_time=network_thinking_time,
            recurrent_activation_function=config["recurrent_activation_function"],
            additive_update=config["additive_update"],
            persistent_observation=persistent_observation,
            feature_transformation_model=None,
        )
        # Select action from the output nodes
        output = network_state[-config["action_dim"] :]
        loss = criterion(label, output)
        eval_loss += loss

    return eval_loss


def fitness_functional(config: dict, graph: meta_ndp):
    def fitness(evolved_parameters: np.array):
        mean_reward = 0

        for _ in range(config["num_growth_evals"]):
            if config["num_growth_evals"] > 1:
                config["seed"] = None

            seed_python_numpy_torch_cuda(config["seed"])

            # init graph
            G = graph.generate_initial_graph(
                network_size=config["network_size"],
                sparsity=config["sparsity"],
                binary_connectivity=config["binary_connectivity"],
                undirected=config["undirected"],
                seed=config["seed"],
            )

            # init network state
            if config["coevolve_initial_embd"]:
                initial_network_state = np.expand_dims(
                    evolved_parameters[:, config["node_embedding_size"]], axis=0
                )
            elif config["shared_initial_embd"] and config["random_initial_embd"]:
                initial_network_state = config["initial_network_state"]
            else:
                initial_network_state = np.random.rand(
                    config["initial_network_size"], config["node_embedding_size"]
                )

            # create growth decision network
            mlp_growth_model = graph.mlp(
                input_dim=config["node_size _growth_model"],
                output_dim=1,
                hidden_layers_dims=config["mlp_growth_hidden_layers_dims"],
                last_layer_activated=config["growth_model_last_layer_activated"],
                activation=torch.nn.Tanh(),
                bias=config["growth_model_bias"],
            )

            mlp_weight_model = graph.mlp(
                input_dim=config["node_embedding_size"] * 2,
                output_dim=config["node_embedding_size"],
                hidden_layers_dims=config["mlp_weight_hidden_layers_dims"],
                last_layer_activated=config["weight_model_last_layer_activated"],
                activation=torch.nn.Tanh(),
                bias=config["weight_model_bias"],
            )

            i1 = config["node_embedding_size"] if config["coevolve_initial_embd"] else 0
            i2 = i1 + config["params_growth_model"]
            i3 = i2 + config["params_weight_model"]

            torch.nn.utils.vector_to_parameters(
                torch.tensor(
                    evolved_parameters[i1:i2], dtype=torch.float64, requires_grad=False
                ),
                mlp_growth_model.parameters(),
            )

            torch.nn.utils.vector_to_parameters(
                torch.tensor(
                    evolved_parameters[i2:i3], dtype=torch.float64, requires_grad=False
                ),
                mlp_weight_model.parameters(),
            )

            network_state = copy.deepcopy(initial_network_state)
            obs_dim_tuple = (config["observation_dim"], config["observation_dim"])

            for growth_cycle_nb in range(config["number_of_growth_cycles"]):
                try:
                    diameter = nx.diameter(G.to_undirected())
                except:
                    diameter = int(np.sqrt(len(G)))

                network_thinking_time = (
                    diameter + config["network_thinking_time_extra_growth"]
                )

                network_state = graph.propagate_features(
                    network_state=network_state,
                    network_thinking_time=network_thinking_time,
                    activation_function=torch.nn.Tanh(),
                    additive_update=True,
                    feature_transformation_model=None,
                    persistent_observation=None,
                )

                if config["node_based_growth"]:
                    embeddings_for_growth_model = network_state

                new_nodes_prediction = graph.predict_new_nodes(
                    model=mlp_growth_model,
                    concat_node_embeddings=embeddings_for_growth_model,
                )

                # TO-DO : complete the add_new_nodes function
                G, network_state = graph.add_new_nodes(
                    config=config, new_nodes_prediction=new_nodes_prediction
                )

                G = graph.update_weights(
                    network_state=network_state, config=config, model=mlp_weight_model
                )

                if config["pruning"]:
                    edges_to_prune = [
                        (a, b)
                        for a, b, attrs in G.edges(data=True)
                        if abs(attrs["weight"]) <= config["pruning_threshold"]
                    ]
                    graph.G.remove_edges_from(edges_to_prune)

            mean_episode_reward = 0
            if len(G) < config["min_network_size"]:
                if config["maximize"]:
                    return len(G) - config["min_network_size"]
                else:
                    return config["min_network_size"] - len(G)

            for _ in range(config["num_episode_evals"]):
                # code for MNIST data evaluation
                seed_env_eval = int(
                    np.random.default_rng(config["env_seed"]).integers(2**32, size=1)[
                        0
                    ]
                )
                episode_reward = mnist_eval(G, config, seed_env_eval)
                mean_episode_reward += episode_reward

            mean_reward += mean_episode_reward / config["num_episode_evals"]

        mean_reward /= config["num_growth_evals"]
        return mean_reward

    return fitness


def main():
    pass
