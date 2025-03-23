import torch
import torch.nn as nn
import math

from models.vit import TransformerBlock, PositionalEmbedding, PatchEmbedding


class EncoderVIT(torch.nn.Module):
    """ Encoder for the VQ-VAE """

    def __init__(self, model_settings):
        super(EncoderVIT, self).__init__()

        self.num_hidden = model_settings["num_hidden"]
        num_residual_hidden = model_settings["num_residual_hidden"]

        self.num_channels = model_settings["num_channels"]
        self.input_shape = model_settings["input_shape"]
        self.embedding_size = model_settings["embedding_dim"]
        self.patch_size = model_settings["patch_size"]

        self.patch_embedding = PatchEmbedding(model_settings)
        self.positional_embedding = PositionalEmbedding(self.patch_embedding.num_patches, model_settings)
        
        self.transformers = nn.Sequential(*[TransformerBlock(model_settings) for i in range(model_settings["transformer_layers"])])
        self.conv_to_vq_size = nn.Conv2d(self.num_hidden, self.embedding_size, stride=1, kernel_size=1)

        
        
    def forward(self, x):
        x = self.patch_embedding(x) 
        pos_embeddings = self.positional_embedding(x)
        x = x + pos_embeddings
        x = self.transformers(x)

        x = x.reshape(x.shape[0], # Reshape to 2d
                (self.input_shape[0]//self.patch_size), 
                (self.input_shape[1]//self.patch_size), 
                self.num_hidden).movedim(-1, 1)
        x = self.conv_to_vq_size(x)
        return x


class Decoder(torch.nn.Module):
    """ Decoder for the VQ-VAE """

    def __init__(self, model_settings):
        embedding_dim = model_settings["embedding_dim"]
        num_hidden = model_settings["num_hidden"]
        num_residual_hidden = model_settings["num_residual_hidden"]
        channels = model_settings["num_channels"]

        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hidden, stride=1, kernel_size=3, padding=1),
            ResidualLayer(num_hidden, num_hidden, num_residual_hidden),
            ResidualLayer(num_hidden, num_hidden, num_residual_hidden),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hidden, num_hidden, stride=2, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hidden, num_hidden//2 ,kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hidden//2, channels ,kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()                                
        )

    def forward(self, x):
        return self.layers(x)

class ResidualLayer(torch.nn.Module):
    """ Residual layer for the VQ-VAE """

    def __init__(self, in_channels, num_hidden, num_residual_hidden):
        super(ResidualLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, num_residual_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_residual_hidden, num_hidden, kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)
        