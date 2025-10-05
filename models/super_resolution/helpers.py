import torch
import yaml
import datetime
import os

from PIL import Image

# --- Helper functions ---

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")

def get_device_auto():
    """Get CUDA if available, else cpu"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model, model_path, device):
    """Load saved model"""
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Model file not found at {model_path}.")
        return

def save_model(model, model_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, extension = os.path.splitext(model_path)
    timestamped_model_path = f"{base_name}_{timestamp}{extension}"
    torch.save(model.state_dict(), timestamped_model_path)
    
def load_img_file(file_path):
    if not os.path.exists(file_path):
        print(f"Input image not found: {file_path}")
        return
    return Image.open(file_path).convert('RGB')
