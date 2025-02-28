import torch
import torchvision.transforms as transforms
import torchvision

import os

from models.vq_vae import VQVAE


def predict(dataset, model):
    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"Device: {device}" + "\n")
    
    img_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    val = torchvision.datasets.ImageFolder("datasets/xray/Data/val/", transform=img_transforms)

    dataloader = torch.utils.data.DataLoader(val, batch_size=256, num_workers=0, shuffle=True)

    
    
    model_settings = {
        "num_channels": 1,
        "input_shape": (256,256),
        "num_hidden": 64,
        "num_residual_hidden": 32,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "commitment_cost": 0.5,
    }
    model = VQVAE(model_settings=model_settings)
    model.load_state_dict(torch.load("models/saved_models/x-ray/model.pt", weights_only=True))
    model.to(device)
    
    
    labels = []
    images = []
    encoder_embeddings = []
    quantized_embeddings = []

    with torch.no_grad():
        for x_test, y_test in dataloader:
            print(x_test.shape)
            images.append(x_test.cpu())
            x_test= x_test.to(device)
            
            # Embeddings just before discretizing them
            x = model.encoder(x_test)
            encoder_embeddings.append(x.cpu())

            # Discrete embeddings and Quantized embeddings
            quantized, vq_loss, discrete_embedding = model.VQ(x)
            quantized_embeddings.append(quantized.cpu())
            labels.append(y_test.cpu())

            del x_test, x, quantized, vq_loss, discrete_embedding
            torch.cuda.empty_cache()
        print("test")
    
    print("test")
    dir = "datasets/xray_embeddings/"
    os.makedirs(dir, exist_ok = True) 

    labels = torch.concat(labels)
    images = torch.concat(images)
    encoder_embeddings = torch.concat(encoder_embeddings)
    quantized_embeddings = torch.concat(quantized_embeddings)
    

    torch.save(labels, dir + "labels.pt")
    torch.save(images, dir + "images_transformed.pt")
    torch.save(encoder_embeddings, dir + "encoded.pt")
    torch.save(quantized_embeddings, dir + "quantized.pt")




            
            
    





if __name__ == "__main__":
    dataset = "x-ray"
    model = "model.pt"
    predict(dataset,model)
    