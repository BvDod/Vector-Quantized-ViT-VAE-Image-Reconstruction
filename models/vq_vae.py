import torch

class VQVAE(torch.nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.VQ = VectorQuantisizer()
        self.decoder = Decoder()
    
    def forward(self, x):
        pass


class Encoder(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class VectorQuantisizer(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class Decoder(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

