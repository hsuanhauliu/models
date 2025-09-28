import os
import numpy as np
from PIL import Image

# Configuration
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
NUM_TRAIN_IMAGES = 50 # Diffusion models benefit from more data
NUM_TEST_IMAGES = 1
# Generate larger images for a more meaningful super-resolution task
IMAGE_SIZE = (128, 128) # height, width

def generate_dummy_image(size, filename):
    """Generates and saves a simple dummy image with random noise."""
    random_array = np.random.rand(size[0], size[1], 3) * 255
    im = Image.fromarray(random_array.astype('uint8')).convert('RGB')
    im.save(filename)

def main():
    """Creates directories and generates dummy images."""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    print(f"Generating {NUM_TRAIN_IMAGES} dummy training images (size: {IMAGE_SIZE})...")
    print("NOTE: For real training, replace these with high-quality images (e.g., from DIV2K).")
    for i in range(NUM_TRAIN_IMAGES):
        img_path = os.path.join(TRAIN_DIR, f'train_dummy_{i+1}.png')
        generate_dummy_image(IMAGE_SIZE, img_path)
    
    print(f"\nGenerating {NUM_TEST_IMAGES} dummy test image...")
    test_img_path = os.path.join(TEST_DIR, 'test_image.png')
    generate_dummy_image(IMAGE_SIZE, test_img_path)
    
    print("\nDummy dataset created successfully!")

if __name__ == '__main__':
    main()

