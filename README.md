# Vision Transformer for CIFAR-10
This repository contains an implementation of a Vision Transformer (ViT) for image classification on the CIFAR-10 dataset using PyTorch.

## About
The Vision Transformer (ViT) model leverages the transformer architecture, originally designed for natural language processing, to perform image classification tasks. The model divides an image into patches, embeds these patches, and processes them through a series of transformer blocks to capture complex patterns and relationships. The final output is a classification of the image into one of the CIFAR-10 classes.

## Key Components:
Patch Embedding: Converts the input image into a sequence of flattened patches.
Transformer Blocks: Consists of multi-head self-attention and feed-forward neural networks.
Classification Head: Outputs the class probabilities for the input image.
## Tech Stack
<table>
  <tr>
    <td><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&amp;logo=PyTorch&amp;logoColor=white"></td>
    <td>PyTorch: Deep learning framework used for building and training the model.</td>
  </tr>
  <tr>
    <td><img alt="Torchvision" src="https://img.shields.io/badge/Torchvision-%23EE4C2C.svg?style=for-the-badge&amp;logo=PyTorch&amp;logoColor=white"></td>
    <td>Provides datasets and image transformations.</td>
  </tr>
  <tr>
    <td><img alt="CUDA" src="https://img.shields.io/badge/CUDA-%23007ACC.svg?style=for-the-badge&amp;logo=NVIDIA&amp;logoColor=white"></td>
    <td>GPU acceleration for faster training.</td>
  </tr>
  <tr>
    <td><img alt="NumPy" src="https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&amp;logo=NumPy&amp;logoColor=white"></td>
    <td>Library for numerical computations.</td>
  </tr>
  <tr>
    <td><img alt="Python" src="https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&amp;logo=Python&amp;logoColor=white"></td>
    <td>Programming language used for implementing the model.</td>
  </tr>
</table>



![GitHub stars](https://img.shields.io/github/stars/mikgrycz/Car-selling-platform?style=social)
![GitHub forks](https://img.shields.io/github/forks/mikgrycz/Car-selling-platform?style=social)
![GitHub issues](https://img.shields.io/github/issues/mikgrycz/Car-selling-platform)
![GitHub license](https://img.shields.io/github/license/mikgrycz/Car-selling-platform)
![Python](https://img.shields.io/badge/python-3.10-blue)
