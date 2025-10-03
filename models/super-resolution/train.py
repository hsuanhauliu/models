import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage, RandomCrop, Lambda
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

from diffusion_model import UNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS, device="cuda")
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# --- Helper Functions ---
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.to(DEVICE))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(DEVICE)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, low_res_img, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, low_res_img)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# --- Sampling functions for logging ---
@torch.no_grad()
def p_sample(model, x, t, t_index, low_res_img):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, low_res_img) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_and_log_images(model, low_res_upscaled, hr_image, epoch, writer):
    model.eval()
    shape = hr_image.shape
    device = next(model.parameters()).device
    
    img = torch.randn(shape, device=device)
    
    for i in reversed(range(0, TIMESTEPS)):
        img = p_sample(model, img, torch.full((shape[0],), i, device=device, dtype=torch.long), i, low_res_upscaled)
    
    # Normalize all images to [0, 1] for grid view
    generated_img = (img.clamp(-1, 1) + 1) / 2
    hr_image_grid = (hr_image.clamp(-1, 1) + 1) / 2
    low_res_grid = (low_res_upscaled.clamp(-1, 1) + 1) / 2
    
    # Log the first image of the batch
    grid = make_grid([low_res_grid[0], generated_img[0], hr_image_grid[0]], nrow=3)
    writer.add_image(f'Epoch {epoch}: Low-Res / Generated / High-Res', grid, epoch)
    model.train()


# --- Dataset ---
class SuperResDataset(Dataset):
    def __init__(self, image_dir, img_size):
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.transform = Compose([
            RandomCrop(img_size),
            ToTensor(), # Scales to [0, 1]
            Lambda(lambda t: (t * 2) - 1) # Scale to [-1, 1]
        ])

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.image_filenames)
