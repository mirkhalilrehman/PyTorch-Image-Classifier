Apologies for the misunderstanding earlier! Here is the entire content in one complete markdown file as requested:

```markdown
# CIFAR-10 Image Classification with Convolutional Neural Network (CNN)

This project demonstrates how to classify CIFAR-10 images using a Convolutional Neural Network (CNN) implemented in PyTorch. The CNN model uses data augmentation, dropout, and multiple convolutional layers to improve generalization and reduce overfitting. The model is trained on a GPU if available, and performance is evaluated on a validation set.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Project Overview

This project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. The goal is to train a CNN to classify images into one of these 10 classes:  
`["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]`

The model is implemented using PyTorch and includes techniques such as data augmentation, dropout, and convolutional layers.

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.8.0 or higher
- NumPy
- Matplotlib

### Install dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

## Dataset

The CIFAR-10 dataset is available through the `torchvision.datasets` module. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is divided into 50,000 training images and 10,000 testing images.

You can download the dataset by running the script. The `datasets.CIFAR10` class will automatically download the dataset and store it in a directory specified in the code.

## Model Architecture

The CNN model consists of the following layers:

- **Convolutional Layers**: Three convolutional layers (`Conv2d`) to extract features from the images.
- **Pooling**: Max-pooling layers (`MaxPool2d`) to reduce the spatial dimensions of the feature maps.
- **Fully Connected Layers**: Two fully connected layers (`Linear`) to classify the image into one of the 10 classes.
- **Dropout**: Applied after the fully connected layers to reduce overfitting.
- **Activation**: ELU (Exponential Linear Unit) activation functions are applied after each convolutional and fully connected layer.

## Training

The model is trained for 40 epochs using the following setup:

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum
- **Learning Rate**: 0.001
- **Batch Size**: 20
- **Data Augmentation**: Random horizontal flip, random rotation

The training and validation loss are printed after each epoch. The model with the lowest validation loss is saved.




```

