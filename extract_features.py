import torch
import torchvision.transforms as transforms
import torchvision

import os

from models.vq_vae import VQVAE
from functions.dataHandling import get_dataset
from torch.utils.data import DataLoader


def extract_embeddings(dataset, model):
    settings = {
        "dataset": "celebA",
        "save_model": True,

        "print_debug": False,
        "example_image_amount": 4,
        "save_reconstructions_first_epoch": True,
        "batch_size": 16,
        "learning_rate": 3e-4, # for x-ray
        "max_epochs": 100000,
        "early_stopping_epochs": 3,

        "model_settings" : {
            "encoder_architecture": "VIT",
            "decoder_architecture": "VIT",
            "num_hidden": 128,
            "num_residual_hidden": 128,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "commitment_cost": 0.25,
            "transformer_layers": 5,
            "attention_heads": 4,
            "patch_size": 8,
        }
    }


    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"Device: {device}" + "\n")
    
    # Load dataset
    train, test, input_shape, channels, train_var = get_dataset(settings["dataset"], print_stats=True, disable_augmentation=True)
    
    dataloader_train = DataLoader(train, batch_size=32, shuffle=True, drop_last=False, pin_memory=False, num_workers=6)
    dataloader_test = DataLoader(test, batch_size=32, pin_memory=False, num_workers=6)

    # Load model
    model_settings = settings["model_settings"]
    model_settings["num_channels"] = channels
    model_settings["input_shape"] = input_shape
   
    model = VQVAE(model_settings=model_settings)
    model.load_state_dict(torch.load("models/saved_models/celebA/model_best(222).pt", weights_only=True))
    model.to(device)
    

    def save_discrete_embeddings(dataloader, model, filename):
        """ Functions that saves the discrete embeddings of the dataset at filename"""

        discrete_embeddings = []

        model.eval()
        with torch.no_grad():
            for x_test, y_test in dataloader:
                x_test= x_test.to(device)
                
                # Embeddings just before discretizing them
                x = model.encoder(x_test)
                # Discrete embeddings and Quantized embeddings
                quantized, vq_loss, discrete_embedding = model.VQ(x)
                
                discrete_embeddings.append(torch.argmax(discrete_embedding.cpu(), dim=1).short())

                # Because we are independently running parts of the model, some data is retained between batches
                del x_test, x, quantized, vq_loss, discrete_embedding
                torch.cuda.empty_cache()
        
        # Save labels, image, and embeddings
        dir = "datasets/celebA_embeddings/"
        os.makedirs(dir, exist_ok = True) 

        discrete_embeddings = torch.concat(discrete_embeddings)
        torch.save(discrete_embeddings, dir + filename)
    
    save_discrete_embeddings(dataloader_train, model, "discrete_train.pt")
    save_discrete_embeddings(dataloader_test, model, "discrete_test.pt")
    

if __name__ == "__main__":
    extract_embeddings()
    