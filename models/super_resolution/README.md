# Conditional Diffusion Model for Super-Resolution

This project is a PyTorch implementation of a simple U-Net based conditional diffusion model for image super-resolution. The model is trained to upscale low-resolution images by generating plausible high-frequency details.

## Dataset

[DIV2K Dataset](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) was used to train and test this model.

## Usage

`config.yaml`: All settings, including image size, learning rate, epochs, and file paths, can be modified in the config.yaml file.

`train.ipynb`: notebook for training the model.

`inference.ipynb`: notebook for running inference.