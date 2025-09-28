import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os
from tqdm import tqdm

from diffusion_model import SimpleUNet
from train import (
    TIMESTEPS, alphas, alphas_cumprod, sqrt_recip_alphas, 
    sqrt_one_minus_alphas_cumprod, posterior_variance, extract
)

# --- Configuration ---
MODEL_PATH = 'diffusion_upscaler.pth'
INPUT_IMAGE_PATH = 'data/test/test_image.png'
OUTPUT_IMAGE_PATH = 'output/upscaled_diffusion.png'
UPSCALE_FACTOR = 2
TARGET_HR_SIZE = 64 # Must match training IMG_SIZE
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Sampling Function ---
@torch.no_grad()
def p_sample(model, x, t, t_index, low_res_img):
    betas_t = extract(torch.linspace(0.0001, 0.02, TIMESTEPS), t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the DDPM paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, low_res_img) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, low_res_img):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device) # Start with pure noise
    imgs = []

    for i in tqdm(reversed(range(0, TIMESTEPS)), desc='Sampling loop', total=TIMESTEPS):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, low_res_img)
        # Optional: save intermediate images
        # if i % 50 == 0:
        #     imgs.append(img.cpu())
    return img

# --- Main Inference ---
def upscale_image(image_path):
    if not os.path.exists(image_path):
        print(f"Input image not found: {image_path}")
        return

    # Load model
    model = SimpleUNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
        return

    # Load and prepare image
    low_res_pil = Image.open(image_path).convert('RGB')
    
    # For inference, the input image is the low-resolution one
    # So we resize it to the expected low-res size
    lr_w, lr_h = TARGET_HR_SIZE // UPSCALE_FACTOR, TARGET_HR_SIZE // UPSCALE_FACTOR
    low_res_pil = low_res_pil.resize((lr_w, lr_h), Image.BICUBIC)

    # Upscale with bicubic for conditioning and comparison
    bicubic_upscaled = low_res_pil.resize((TARGET_HR_SIZE, TARGET_HR_SIZE), Image.BICUBIC)
    
    to_tensor = ToTensor()
    low_res_tensor = to_tensor(bicubic_upscaled).unsqueeze(0).to(DEVICE)

    # Run the diffusion sampling process
    print("Starting diffusion sampling...")
    output_tensor = p_sample_loop(model, 
                                  shape=(1, 3, TARGET_HR_SIZE, TARGET_HR_SIZE), 
                                  low_res_img=low_res_tensor)

    # Process and save output
    output_tensor = (output_tensor.clamp(-1, 1) + 1) / 2 # Normalize to [0, 1] for saving
    to_pil = ToPILImage()
    output_pil = to_pil(output_tensor.squeeze(0).cpu())
    
    os.makedirs('output', exist_ok=True)
    bicubic_upscaled.save(OUTPUT_IMAGE_PATH.replace('.png', '_bicubic.png'))
    output_pil.save(OUTPUT_IMAGE_PATH)

    print(f"Bicubic upscaled image saved to {OUTPUT_IMAGE_PATH.replace('.png', '_bicubic.png')}")
    print(f"Diffusion upscaled image saved to {OUTPUT_IMAGE_PATH}")

if __name__ == '__main__':
    upscale_image(INPUT_IMAGE_PATH)
