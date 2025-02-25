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
        quantized, vq_loss, discrete_embedding = self.VQ(x)
        x = self.decoder(quantized)
        return x, vq_loss


class Encoder(torch.nn.Module):
    def __init__(self, input_shape=(28,28), num_hidden=64, num_residual_hidden=16, embedding_dim=32):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, num_hidden//2, stride=2, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hidden//2, num_hidden, stride=2, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, stride=1, kernel_size=3, padding=1),
            # ResidualLayer(num_hidden, num_hidden, num_residual_hidden),
            # ResidualLayer(num_hidden, num_hidden, num_residual_hidden),   
            nn.ReLU(),       
            nn.Conv2d(num_hidden, embedding_dim, stride=1, kernel_size=1), # Make correct size for VQ
        )
        
        
    def forward(self, x):
        return self.layers(x)

class VectorQuantisizer(torch.nn.Module):
    def __init__(self, num_embeddings=512, embeding_dim=32, commitment_cost=0.26):
        super(VectorQuantisizer, self).__init__()
        self.embedding_dim = embeding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        max = math.sqrt(3/self.num_embeddings) # See Tensorflowv1:UniformUnitScaling code, amounts to this.
        nn.init.uniform_(self.embeddings.weight, -max, max)

    def forward(self, x):
        x = torch.movedim(x, 1, -1) # Move channel dimension to last dim
        input = x
        x = torch.flatten(x, start_dim=0, end_dim=-2) # Flatten all but batch dimension

        distances = (
        torch.sum(x**2, dim=-1, keepdim=True)  # (N, K, 1)
        + torch.sum(self.embeddings.weight**2, dim=-1)  # (E,)
        - 2 * torch.matmul(x, self.embeddings.weight.T)  # (N, K, E)
    )
        embedding_indexes = torch.argmin(distances, dim=-1) # Get the index of the closest embedding vector
        discrete_embedding = nn.functional.one_hot(embedding_indexes, num_classes=self.num_embeddings)
        
        discrete_embedding = torch.reshape(discrete_embedding, input.shape[:-1] + (self.num_embeddings,)) # Reshape to original shape
        quantized = discrete_embedding.float() @ self.embeddings.weight
        # shape = (n_batch, embeddings_size)

        embeddings_loss = torch.mean((quantized - input.detach())**2) # loss is difference between encoder output and quantisized
        commitment_loss = self.commitment_cost * torch.mean((quantized.detach() - input)**2)
        vq_loss = embeddings_loss + commitment_loss
    	
        quantized = input + (quantized - input).detach() # to move gradients from decoder to encoder
        
        #Change back to original shape
        quantized = torch.movedim(quantized, -1, 1)
        discrete_embedding = torch.movedim(discrete_embedding, -1, 1)
        
        return quantized, vq_loss, discrete_embedding

        

class Decoder(torch.nn.Module):
    def __init__(self, num_hidden=64, num_residual_hidden=16, embedding_dim=32):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hidden, stride=1, kernel_size=3, padding=1),
            #ResidualLayer(num_hidden, num_hidden, num_residual_hidden),
            #ResidualLayer(num_hidden, num_hidden, num_residual_hidden),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hidden, num_hidden//2 ,kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hidden//2, 1 ,kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()                                
        )

    def forward(self, x):
        return self.layers(x)

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
        
