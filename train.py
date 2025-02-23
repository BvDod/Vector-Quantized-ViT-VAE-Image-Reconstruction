from functions.dataHandling import get_dataset
from functions.Visualize import plot_grid_samples
from models.vq_vae import Encoder, VQVAE

import torch
from torch.utils.data import DataLoader

from torchinfo import summary





def train(settings):
    print(str(settings) + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + "\n")

    train, test = get_dataset(settings["dataset"], print_stats=True)
    
    if settings["print_samples"]:
        plot_grid_samples(train)
    
    dataloader_train = DataLoader(train, batch_size=64, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(test, batch_size=64)

    print(summary(Encoder(), input_size=(64, 1, 28,28)))
    model = VQVAE()

    for x_train, y_train in dataloader_train:
        print(x_train.shape)
        pred = model(x_train)
        exit()
    








if __name__ == "__main__":
    train_settings = {
        "dataset" : "MNIST",
        "print_samples": False
    }
    train(train_settings)