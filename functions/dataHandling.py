""" Contains all functions regarding file/dataset handling """

import torch
import torchvision
import torchvision.transforms as transforms

DATASETS = {
    "MNIST": torchvision.datasets.MNIST
}


""" Retrieves dataset with name dataset name from disk and returns it"""
def get_dataset(dataset_name: str, print_stats = False):

    if not dataset_name in DATASETS:
        print("ERROR: invalid dataset name: {dataset_name}")
    
    img_transforms = transforms.Compose([transforms.ToTensor()])

    train = DATASETS[dataset_name](root="./datasets/", train = True, transform=img_transforms)
    test = DATASETS[dataset_name](root="./datasets/", train = False, transform=img_transforms)

    if print_stats:
        print(f"Training dataset shape: {train.data.shape}")
        print(f"Test dataset shape: {train.data.shape}")
        print(f"Unique training labels: {train.targets.unique(return_counts=True)}")
        print(f"Training dtype: {train.data.dtype}")
        print("\n")

    return train, test

