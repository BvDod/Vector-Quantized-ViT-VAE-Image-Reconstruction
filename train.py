# %%
from functions.dataHandling import get_dataset
from functions.visualize import plot_grid_samples, plot_grid_samples_tensor
from models.vq_vae import Encoder, VQVAE
import numpy as np

import torch
from torch.utils.data import DataLoader

from torchinfo import summary


def train(settings):
        
    print(str(settings) + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + "\n")

    train, test = get_dataset(settings["dataset"], print_stats=True)
    train_var = (train.data /255.0).var()
    
    print(train_var)
    if settings["print_samples"]:
        plot_grid_samples(train)

    dataloader_train = DataLoader(train, batch_size=32, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(test, batch_size=32, num_workers=4, pin_memory=True)



    print(summary(VQVAE(), input_size=(32, 1, 28,28)))

    learning_rate = 3e-4
    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    model.train()

    train_losses = []
    test_losses = []
    for epoch in range(100):
        train_losses_epoch = []
        print(f"Epoch: {epoch}/100")
        for x_train, y_train in dataloader_train:
            x_train = x_train.to(device)
            pred, vq_loss = model(x_train)
            loss = (torch.nn.functional.mse_loss(x_train, pred) / train_var) + vq_loss
            print(loss, vq_loss)
            loss.backward()
            optimizer.step()
            train_losses_epoch.append(loss.item())
            optimizer.zero_grad()
        print(f"train loss: {sum(train_losses_epoch) / len(train_losses_epoch)}")
        train_losses.append(sum(train_losses_epoch) / len(train_losses_epoch))
        if epoch % 3 == 0:
            with torch.no_grad():
                test_losses_epoch = []
                for x_test, y_test in dataloader_test:
                    x_test = x_test.to(device)
                    pred, vq_loss = model(x_test)
                    loss = (torch.nn.functional.mse_loss(x_test, pred) / train_var) + vq_loss
                    test_losses_epoch.append(loss.item())
                print(f"Test loss: {sum(test_losses_epoch) / len(test_losses_epoch)}")
                plot_grid_samples_tensor(x_test)
                plot_grid_samples_tensor(pred)
                test_losses.append(sum(test_losses_epoch) / len(test_losses_epoch))

    import matplotlib.pyplot as plt

    plt.plot(train_losses)
    plt.show()

    plt.plot

if __name__ == "__main__":
    settings = {
        "dataset": "MNIST",
        "print_samples": False
    }
    train(settings)