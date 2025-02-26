""" Contains all functions regarding file/dataset handling """

import torchvision
import torchvision.transforms as transforms
from functions.customDatasets import xrayDataset


""" Retrieves dataset with name dataset name from disk and returns it"""
def get_dataset(dataset_name: str, print_stats = False):

    if dataset_name == "x-ray":
        img_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    else:   
        img_transforms = transforms.Compose([                                     
            transforms.ToTensor(),]
        )

    if dataset_name == "MNIST":
        train = torchvision.datasets.MNIST(root="./datasets/", train=True, transform=img_transforms,  download=True)
        test = torchvision.datasets.MNIST(root="./datasets/", train=False, transform=img_transforms, download=True)
        input_shape = (28, 28)
        channels = 1

    elif dataset_name == "SLT10":
        train = torchvision.datasets.STL10(root="./datasets/", split="train", transform=img_transforms, download=True)
        test = torchvision.datasets.STL10(root="./datasets/", split="test", transform=img_transforms, download=True)
        input_shape = (96, 96)
        channels = 3
    
    elif dataset_name == "x-ray":
        train = xrayDataset("./datasets/xray/", train=True, transform=img_transforms)
        test = xrayDataset("./datasets/xray/", train=False, transform=img_transforms) 
        input_shape = (256, 256)
        channels = 1

    if print_stats:
        print(f"Training dataset shape: {train[0][0].shape}")
        print(f"Test dataset shape: {test[0][0].shape}")
        print("\n")

    return train, test, input_shape, channels

