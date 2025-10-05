import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Lambda, Compose, ToPILImage, Resize, InterpolationMode
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import yaml

from diffusion_model import UNet
import helpers

# --- Diffusion Forward Process Helpers ---
def linear_beta_schedule(beta_start, beta_end, timesteps):
    """Creates an array of timesteps values that increase linearly.
    
    Each beta value represents the variance (or "amount") of noise to add at that specific timestep. Early steps have small betas (add little noise), and later steps have larger betas."""
    return torch.linspace(beta_start, beta_end, timesteps)

def get_diffusion_variables(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = linear_beta_schedule(beta_start, beta_end, timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return (betas, alphas, alphas_cumprod, alphas_cumprod_prev, 
            sqrt_recip_alphas, sqrt_alphas_cumprod, 
            sqrt_one_minus_alphas_cumprod, posterior_variance)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """Perform forward diffusion."""
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Extract values for the given timesteps
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    # Return noisy image and the noise itself
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

def extract(a, t, x_shape):
    """Select the correct pre-calculated values for a given batch of timesteps."""
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# --- Dataset ---
class SuperResolutionDataset(Dataset):
    def __init__(self, root_dir, img_size, upscale_factor):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.hr_transform = Compose([
            Resize((img_size, img_size), InterpolationMode.BICUBIC),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)
        ])
        self.lr_transform = Compose([
            Resize((img_size // upscale_factor, img_size // upscale_factor), InterpolationMode.BICUBIC),
            Resize((img_size, img_size), InterpolationMode.BICUBIC), # Upscale for conditioning
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        hr_image = self.hr_transform(image)
        lr_image = self.lr_transform(image)
        return hr_image, lr_image

# --- Training ---
def train_pipeline(config: dict):
    # Setup
    device = helpers.get_device_auto()
    print(f"Using device: {device}")
    
    writer = SummaryWriter(config['output']['log_dir'])
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # Data
    dataset = SuperResolutionDataset(
        root_dir=config['data']['train_dir'],
        img_size=config['data']['img_size'],
        upscale_factor=config['data']['upscale_factor']
    )
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    
    # Model
    model = UNet(
        in_channels=config['model']['in_channels'], 
        out_channels=config['model']['out_channels'],
    ).to(device)
    model_path = config['output']['model_path']
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    # Diffusion variables
    timesteps = config['model']['timesteps']
    _, _, _, _, _, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, _ = get_diffusion_variables(timesteps)
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)

    # Training loop
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (hr_images, lr_images) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)
            
            t = torch.randint(0, timesteps, (hr_images.shape[0],), device=device).long()
            
            noisy_hr, noise = q_sample(hr_images, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            
            predicted_noise = model(noisy_hr, t, lr_images)
            
            loss = criterion(noise, predicted_noise)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

        # Simplified TensorBoard logging, can be expanded
        if (epoch + 1) % config['training']['save_image_every'] == 0:
            print(f"Epoch {epoch+1}: Loss: {loss.item()}")

        if (epoch + 1) % config['training']['save_model_every'] == 0:
            print(f"Saved model at epoch: {epoch+1}")
            helpers.save_model(model, model_path)

    writer.close()
    print(f"Training complete.")

if __name__ == '__main__':
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        train.train_pipeline(config)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure it is in the root directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
