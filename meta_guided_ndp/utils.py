import numpy as np
import torch
import torchvision


def x0_sampling(dist, num_parameters):
    if dist == "U[0,1]":
        return np.random.rand(num_parameters)
    elif dist == "N(0,1)":
        return np.random.randn(num_parameters)
    elif dist == "U[-1,1]":
        return 2 * np.random.rand(num_parameters) - 1
    else:
        raise ValueError("Unknown distribution for x0")


def seed_python_numpy_torch_cuda(seed):
    pass


def mnist_data_loader():
    mnist_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    mnist_loader = torch.utils.data.DataLoader(
        dataset=mnist_data, batch_size=32, shuffle=True
    )
    return mnist_loader
