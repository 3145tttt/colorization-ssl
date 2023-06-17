import torch
import torch.nn as nn

from diffusers.models.vae import Encoder, Decoder

class Autoencoder(nn.Module):
    
    def __init__(
        self,
        in_channels = 1,
        out_channels = 3,
        down_block_types = ["DownEncoderBlock2D"] * 3,
        up_block_types = ["UpDecoderBlock2D"] * 3,
        block_out_channels = (32, 64, 64),
        layers_per_block = 2,
        act_fn = "silu",
        latent_channels = 4,
        norm_num_groups = 32
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )
        
        self.quant_conv = nn.Conv2d(2 * latent_channels, latent_channels, 1)
        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.quant_conv(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return torch.tanh(z)