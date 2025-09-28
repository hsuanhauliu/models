import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Module for generating sinusoidal position embeddings for timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """A basic building block for the U-Net."""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    """
    A simplified U-Net architecture for the diffusion model.
    It predicts the noise added to an image at a specific timestep.
    """
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        image_channels = in_channels
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) \
                    for i in range(len(down_channels)-1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    def forward(self, x, timestep, low_res_img):
        # Concatenate noisy image with low-resolution conditioning image
        x = torch.cat((x, low_res_img), dim=1)
        
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

if __name__ == '__main__':
    # Example of how to instantiate the model
    model = SimpleUNet()
    print("Diffusion U-Net Model Architecture:")
    # print(model) # Printing the full model can be very long

    # Create dummy inputs to test the forward pass
    batch_size = 4
    img_size = 64
    noisy_image = torch.randn(batch_size, 3, img_size, img_size)
    low_res_image_upscaled = torch.randn(batch_size, 3, img_size, img_size)
    timesteps = torch.randint(0, 1000, (batch_size,)).long()

    # The UNet takes the noisy image and the upscaled low-res image as input
    predicted_noise = model(noisy_image, timesteps, low_res_image_upscaled)
    
    print(f"\nInput noisy image shape: {noisy_image.shape}")
    print(f"Input low-res shape: {low_res_image_upscaled.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Output predicted noise shape: {predicted_noise.shape}")
