from functions.dataHandling import get_dataset
from functions.Visualize import plot_grid_samples

import torch
from torch.utils.data import DataLoader






def train(settings):
    print(str(settings) + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + "\n")

    train, test = get_dataset(settings["dataset"], print_stats=True)
    
    if settings["print_samples"]:
        plot_grid_samples(train)
    
    dataloader_train = DataLoader(train, batch_size=64, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(test, batch_size=64)
    








if __name__ == "__main__":
    train_settings = {
        "dataset" : "MNIST",
        "print_samples": True
    }
    train(train_settings)