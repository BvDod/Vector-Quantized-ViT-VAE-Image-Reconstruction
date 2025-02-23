import torch
from torchvision import datasets, transforms

# Download MNIST
train_dataset = datasets.MNIST(root='./datasets', train=True, download=True)
test_dataset = datasets.MNIST(root='./datasets', train=False, download=True)