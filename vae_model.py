
import torch
from torch import nn, optim
from itertools import pairwise
from math import floor, prod, sqrt

class GenerativeVAE(nn.Module):
    def __init__(self, input_shape, latent_dim, device):
        super().__init__() # subclass of neural network module

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.device = device
        self.sigma = torch.load("sigma_vae3.pt").to(device)

        self.encoder_stem =  nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(5,5), stride=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5,5), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Flatten(), 
        )
        
        self.encoder_mean = nn.Linear(in_features=27552, out_features=latent_dim)
        self.encoder_logvar = nn.Linear(in_features=27552, out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features = latent_dim, out_features=27552),

            nn.Unflatten(1, (16, 123, 14)),

            #  nn.Conv2d(stride=1): slides one pixel at a time, with no padding
            #  nn.ConvTranspose2d(stride=1): kernel contributions extend the spatial dimensions rather than shrinking them 
            #        -> this is neccassary for reproducing the correct input_shape


            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(5,5), stride=2, output_padding=(0, 1)), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(5,5), stride=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=(5,5), stride=2), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=(5,5), stride=1),
    
            nn.Sigmoid()
        )

    # input image -> point in latent space
    def encode(self, x):
        enc_stem = self.encoder_stem(x)
        mean = self.encoder_mean(enc_stem)
        logvar = self.encoder_logvar(enc_stem)
        stdev = torch.exp(0.5*logvar)
        norm_sample = torch.randn_like(stdev)
        latent_sample = mean + norm_sample * stdev
        return latent_sample, mean, logvar

    # point in latent space -> reconstructed image
    def decode(self, latent_point):
        decoded = self.decoder(latent_point)
        noise = torch.randn_like(decoded)
        decoded = (decoded + 200 * self.sigma * noise).clamp(0, 1)
        return decoded 

    # input image -> reconstructed image
    def forward(self, x):
        latent_point, mean, logvar = self.encode(x)
        return self.decode(latent_point), mean, logvar
    
    def print_layer_shapes(self):
        x = torch.randn(self.input_shape)  # Example input batch
        print(f"Input shape: {x.shape}")
        print("Encoder stem:")
        for name, layer in self.encoder_stem.named_children():
            x = layer(x)
            if layer.__class__.__name__ != "ReLU":
                print(f"Layer: {name} ({layer.__class__.__name__}), "
                    f"Output shape: {x.shape}")
            
        z = torch.randn(1, self.latent_dim)
        print("Decoder:")
        for name, layer in self.decoder.named_children():
            z = layer(z)
            if layer.__class__.__name__ != "ReLU":
                print(f"Layer: {name} ({layer.__class__.__name__}), "
                    f"Output shape: {z.shape}")
            
        
class StridedConvVAE(nn.Module):
    def __init__(self, input_shape, latent_dim, kernel_sizes, channels, strides, device):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.device = device

        in_dims = (input_shape[0],) + channels
        map_dims = list(input_shape[1:])
        output_paddings = []
        self.encoder_stem = nn.Sequential()
        for layer, ((in_channels, out_channels), kernel_size, stride) in enumerate(zip(pairwise(in_dims), kernel_sizes, strides)):
            pad = [0, 0]
            for idx, dim in enumerate(map_dims):
                map_dims[idx] = (dim - self.pair_value(kernel_size, idx)) // self.pair_value(stride, idx) + 1
                # Keep track of how much of the feature map is not processed by the kernel,
                # such that we can add it back in the decoder to end up with the same input shape
                pad[idx] = (dim - self.pair_value(kernel_size, idx)) % self.pair_value(stride, idx)
            output_paddings.append(pad)
            self.encoder_stem.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=True))
            self.encoder_stem.append(nn.ReLU(True))
        self.encoder_stem.append(nn.Flatten())
        conv_shape = [channels[-1], *map_dims]
        num_units = prod(conv_shape)

        self.encoder_mean = nn.Linear(num_units, latent_dim, bias=True)
        self.encoder_logvar = nn.Linear(num_units, latent_dim, bias=True)

        self.decoder = nn.Sequential()
        self.decoder.append(nn.Linear(latent_dim, num_units, bias=True))
        self.decoder.append(nn.Unflatten(1, conv_shape))
        for (in_channels, out_channels), kernel_size, stride, output_padding in zip(pairwise(in_dims[:0:-1]), kernel_sizes[:0:-1], strides[:0:-1], output_paddings[:0:-1]):
            self.decoder.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, output_padding=output_padding, bias=True))
            self.decoder.append(nn.ReLU(True))
        self.decoder.append(nn.ConvTranspose2d(in_dims[1], in_dims[0], kernel_sizes[0], strides[0], output_padding=output_paddings[0], bias=True))
        self.decoder.append(nn.Sigmoid())

    # Handle both `kernel_size=5` and `kernel_size=(5,5)`
    def pair_value(self, value, dim_idx):
        return value if isinstance(value, (int, str, type(None))) else value[dim_idx]

            # input image -> point in latent space
    def encode(self, x):
        enc_stem = self.encoder_stem(x)

        mean = self.encoder_mean(enc_stem)
        logvar = self.encoder_logvar(enc_stem)
        stdev = torch.exp(0.5*logvar)
        norm_sample = torch.randn_like(stdev)
        latent_sample = mean + norm_sample * stdev
        return latent_sample, mean, logvar

    # point in latent space -> reconstructed image
    def decode(self, latent_point):
        return self.decoder(latent_point)

    # input image -> reconstructed image
    def forward(self, x):
        latent_point, mean, logvar = self.encode(x)
        return self.decode(latent_point), mean, logvar
    
    def print_layer_shapes(self):
        x = torch.randn(self.input_shape, device=self.device)  # Example input batch
        print(f"Input shape: {x.shape}")
        print("Encoder stem:")
        for name, layer in self.encoder_stem.named_children():
            x = layer(x)
            if layer.__class__.__name__ != "ReLU":
                print(f"Layer: {name} ({layer.__class__.__name__}), "
                    f"Output shape: {x.shape}")
            
        z = torch.randn(1, self.latent_dim, device=self.device)
        print("Decoder:")
        for name, layer in self.decoder.named_children():
            z = layer(z)
            if layer.__class__.__name__ != "ReLU":
                print(f"Layer: {name} ({layer.__class__.__name__}), "
                    f"Output shape: {z.shape}")