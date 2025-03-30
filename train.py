# %%
from functions.dataHandling import get_dataset
from functions.visualize import plot_grid_samples_tensor
from models.vq_vae import VQVAE

import torch
from torch.utils.data import DataLoader

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image as im 

import numpy as np

def train_vq_vae(settings):

    # Tensorboard for logging
    writer = SummaryWriter()

    # Print settings and info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(str(settings))
    print(f"Device: {device}" + "\n")

    # Loading dataset
    train, test, input_shape, channels, train_var = get_dataset(settings["dataset"], print_stats=True)
    
    dataloader_train = DataLoader(train, batch_size=settings["batch_size"], shuffle=True, drop_last=True, pin_memory=False, num_workers=6)
    dataloader_test = DataLoader(test, batch_size=32, pin_memory=False, num_workers=6)

    # Setting up model
    model_settings = settings["model_settings"]
    model_settings["num_channels"] = channels
    model_settings["input_shape"] = input_shape
    model = VQVAE(model_settings).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])

    scaler = torch.amp.GradScaler("cuda" ,enabled=True)

    # Training loop
    train_losses, test_losses = [], []
    best_test_loss = float("inf")
    for epoch in range(settings["max_epochs"]):
        train_losses_epoch = []
        print(f"Epoch: {epoch}/{settings["max_epochs"]}")
        
        # Training
        model.train()
        for batch_i, (x_train, _) in enumerate(dataloader_train):
            x_train = x_train.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                pred, vq_loss = model(x_train)
                mse = torch.nn.functional.mse_loss(x_train, pred)
                loss = (mse / train_var) + vq_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses_epoch.append(loss.item())
            optimizer.zero_grad()

            '''
            if (batch_i + 1) % 100 == 0:
                writer.add_scalar("Loss/train (epochs)", sum(train_losses_epoch[-100:])/100, epoch*len(dataloader_train) + batch_i)
                
                with torch.no_grad():
                    x_test, y_test = next(iter(dataloader_test))
                    x_test = x_test.to(device)
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                        pred, vq_loss = model(x_test)
                    grid = plot_grid_samples_tensor(pred[:settings["example_image_amount"]])
                    writer.add_image("Epoch1 reconstructions", grid, batch_i)
            '''
            """
            # Save reconstructions sub-epoch level for first epoch
            if (settings["save_reconstructions_first_epoch"] and (batch_i % 20 == 0) and epoch == 0):
                for x_test, y_test in dataloader_test:
                    pass
                x_test = x_test.to(device)
                pred, vq_loss = model(x_test)
                grid = plot_grid_samples_tensor(pred[:settings["example_image_amount"]])
                writer.add_image("Epoch1 reconstructions", grid, batch_i)
            """
        print(f"Train loss: {sum(train_losses_epoch) / len(train_losses_epoch)}")
        train_losses.append(sum(train_losses_epoch) / len(train_losses_epoch))
        writer.add_scalar("Loss/train", train_losses[-1], epoch)
        
        """
        #  Early stopping
        epoch_delta = settings["early_stopping_epochs"]
        if len(train_losses) > epoch_delta and max(train_losses[-epoch_delta:-1]) < train_losses[-1]:
            print("Early stopping")
            break"
        """
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_losses_epoch = []
            test_mse_epoch = []
            for x_test, y_test in dataloader_test:
                x_test = x_test.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                    pred, vq_loss = model(x_test)
                    mse = torch.nn.functional.mse_loss(x_test, pred)
                    loss = (mse / train_var) + vq_loss
                test_losses_epoch.append(loss.item())
                test_mse_epoch.append(mse.item())
            
            if epoch == 0: # Save target
                grid = plot_grid_samples_tensor(x_test[:settings["example_image_amount"]].cpu())
                writer.add_image("Original", grid, epoch)
            # Save reconstruction
            grid = plot_grid_samples_tensor(pred[:settings["example_image_amount"]].cpu())
            writer.add_image("Reconstruction", grid, epoch)
            
            foldername = "reconstructions/"
            grid = grid.movedim(0,-1)
            image = im.fromarray((grid.cpu().numpy() * 255).astype(np.uint8))
            Path(foldername).mkdir(parents=True, exist_ok=True)
            image.save(f"{foldername}{epoch}.png")

            print(f"Test loss: {sum(test_losses_epoch) / len(test_losses_epoch)}")
            test_losses.append(sum(test_losses_epoch) / len(test_losses_epoch))
            writer.add_scalar("Loss/test", test_losses[-1], epoch)
            writer.add_scalar("MSE/test", sum(test_mse_epoch) / len(test_mse_epoch), epoch)

        if settings["save_model"]:
            import os
            path = f"models/saved_models/{settings["dataset"]}/"
            os.makedirs(path, exist_ok = True) 
            torch.save(model.state_dict(), path + "model_latest.pt")
            if test_losses[-1] < best_test_loss:
                best_test_loss = test_losses[-1]
                torch.save(model.state_dict(), path + f"model_best({epoch}).pt")

if __name__ == "__main__":
    settings = {
        "dataset": "SLT10",
        # "dataset": "MNIST",
        # "dataset": "x-ray",

        "print_debug": False,
        "example_image_amount": 8,
        "save_reconstructions_first_epoch": True,
        "batch_size": 32,
        # "learning_rate": 1e-3, # for x-ray
        # "learning_rate": 1e-4, # for Mnsist
        "learning_rate": 2e-3, # for SLT>?
        "max_epochs": 100,
        "early_stopping_epochs": 50,

        "model_settings" : {
            "num_hidden": 64,
            "num_residual_hidden": 32,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "commitment_cost": 0.5,
            #"commitment_cost": 1 # for mnist
        }
    }
    train_vq_vae(settings)