# Image Utils

from io import BytesIO
import numpy as np
import requests

def load_img(path: str) -> Image:
    """Load an image from disk.
    
    Example:
    > load_img("test_imgs/chess_1.png")
    """
    return Image.open(path, 'r')

def download_img(url: str) -> Image:
    """Download an image from the url.
    
    Example:
    > download_img("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def generate_noise_img(length: int, width: int=0, channel: int=3) -> Image:
    """Generate a noise Pillow image.
    
    Example:
    > generate_noise_img(500, 500) -> 500x500x3 image
    """
    if length == 0:
        raise ValueError("length cannot be 0")
    if width == 0:
        width = length
    np_img = np.random.randint(255, size=(length, width, channel), dtype="uint8")
    return Image.fromarray(np_img)


if __name__ == '__main__':
    download_img("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")