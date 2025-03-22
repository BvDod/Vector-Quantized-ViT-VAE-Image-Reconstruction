""" Contains all functions regarding file/dataset handling """

import torchvision
import torchvision.transforms as transforms
from functions.customDatasets import xrayDataset


def get_dataset(dataset_name: str, print_stats = False):
    """ Retrieves dataset with name dataset name from disk and returns it"""

    if dataset_name == "x-ray":
        img_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    elif dataset_name == "celebA":
        img_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
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
        var = (train.data /255.0).var()
    

    elif dataset_name == "SLT10":
        train = torchvision.datasets.STL10(root="./datasets/", split="train", transform=img_transforms, download=True)
        test = torchvision.datasets.STL10(root="./datasets/", split="test", transform=img_transforms, download=True)
        input_shape = (96, 96)
        channels = 3
        var = (train.data /255.0).var()
    
    elif dataset_name == "x-ray":
        train = xrayDataset("./datasets/xray/", train=True, transform=img_transforms)
        test = xrayDataset("./datasets/xray/", train=False, transform=img_transforms) 
        input_shape = (256, 256)
        var = 0.1102 # Precomputed
        channels = 1
    
    if dataset_name == "celebA":
        train = torchvision.datasets.ImageFolder("./datasets/celeba_hq/train/", transform=img_transforms)
        test = torchvision.datasets.ImageFolder("./datasets/celeba_hq/val/", transform=img_transforms)
        input_shape = (512, 512)
        channels = 3
        var = 0.0692 # Precomputed

    if print_stats:
        print(f"Training dataset shape: {train[0][0].shape}, samples = {len(train)}")
        print(f"Test dataset shape: {test[0][0].shape}, samples = {len(test)}")
        print("\n")

    return train, test, input_shape, channels, var



def get_variance(dataset):
    """ Returns the variance of the dataset , pretty slow because of no batching"""
    var = 0
    for image in dataset:
        var += image[0].var()
    return var / len(dataset)