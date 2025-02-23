import torch
import torch.nn as nn
import math


class VQVAE(torch.nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.encoder = Encoder()
        self.VQ = VectorQuantisizer()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.VQ(x)


class Encoder(torch.nn.Module):
    def __init__(self, input_shape=(28,28), num_hidden=128, num_residual_hidden=32, embedding_dim=64):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, num_hidden//2, stride=2, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hidden//2, num_hidden, stride=2, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, stride=1, kernel_size=3, padding=1),
            ResidualLayer(num_hidden, num_hidden, num_residual_hidden),
            ResidualLayer(num_hidden, num_hidden, num_residual_hidden),   
            nn.ReLU(),       
            nn.Conv2d(num_hidden, embedding_dim, stride=1, kernel_size=1), # Make correct size for VQ
        )
        
        
    def forward(self, x):
        return self.layers(x)

class VectorQuantisizer(torch.nn.Module):
    def __init__(self, num_embeddings=512, embeding_dim=64, commitment_cost=0.25):
        super(VectorQuantisizer, self).__init__()
        self.embedding_dim = embeding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        max = math.sqrt(3/self.num_embeddings) # See Tensorflowv1:UniformUnitScaling code, amounts to this.
        nn.init.uniform_(self.embeddings.weight, -max, max)

    def forward(self, x):
        x = torch.movedim(x, 1, -1) # Move channel dimension to last dim
        input_shape = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=-2) # Flatten all but batch dimension

        distances = torch.cdist(x, self.embeddings.weight, p=2)**2 # det distance from every hidden vector to every embedding vector
        embedding_indexes = torch.argmin(distances, dim=-1) # Get the index of the closest embedding vector
        discrete_embedding = nn.functional.one_hot(embedding_indexes, num_classes=self.num_embeddings)
        
        discrete_embedding = torch.reshape(discrete_embedding, input_shape[:-1] + (self.num_embeddings,)) # Reshape to original shape
        quatisized = discrete_embedding.float() @ self.embeddings.weight
        print(quatisized.shape)
        # shape = (n_batch, embeddings_size)
        # TODO: Implement losses, and sum input to output
        

class Decoder(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class ResidualLayer(torch.nn.Module):
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
        
