# Fashion MNIST Classification with Convolutional Neural Networks (CNNs)

This project demonstrates image classification on the Fashion MNIST dataset using convolutional neural networks (CNNs) built with TensorFlow and Keras. The project explores improvements in model generalization and performance by comparing an initial NN model with an optimized CNN version that incorporates regularization techniques.

## Project Overview

The goal of this project is to classify images from the Fashion MNIST dataset.

### Models

1. **Initial Model**:
   - Basic NN structure with several convolutional and pooling layers.
   - No regularization applied, leading to potential overfitting.

2. **Optimized Model**:
   - Similar CNN architecture but enhanced with **dropout layers** after each convolutional and dense layer.
   - **Early stopping** added to halt training when validation loss stops improving, reducing overfitting.
   - Data augmentation applied (optional) to enhance generalization on the training data.

## Model Architecture

Both models share a similar architecture consisting of convolutional and pooling layers followed by fully connected layers. The optimized model differs by incorporating dropout for regularization.

| Layer Type          | Output Shape            |
|---------------------|-------------------------|
| Conv2D              | (None, 26, 26, 32)      |
| MaxPooling2D        | (None, 13, 13, 32)      |
| Dropout (optional)  | (None, 13, 13, 32)      |
| Conv2D              | (None, 11, 11, 64)      |
| MaxPooling2D        | (None, 5, 5, 64)        |
| Dropout (optional)  | (None, 5, 5, 64)        |
| Conv2D              | (None, 3, 3, 128)       |
| Flatten             | (None, 1152)            |
| Dense               | (None, 128)             |
| Dropout (optional)  | (None, 128)             |
| Dense (Output)      | (None, 10)              |

## Training Results

### Initial Model

| Metric             | Value        |
|--------------------|--------------|
| Training Accuracy  | 92.46%       |
| Training Loss      | 0.1967       |
| Validation Accuracy| 88.54%       |
| Validation Loss    | 0.3538       |

### Optimized Model (with Dropout and Early Stopping)

| Metric             | Value        |
|--------------------|--------------|
| Training Accuracy  | 86.97%       |
| Training Loss      | 0.3559       |
| Validation Accuracy| 89.02%       |
| Validation Loss    | 0.2923       |

## Key Findings

- **Improved Generalization**: The optimized model achieves a higher and more stable validation accuracy (89.02%) compared to the initial model.
- **Reduced Overfitting**: The initial model shows signs of overfitting with a high training accuracy (92.46%) and a notable gap between training and validation performance. The optimized model, with dropout and early stopping, has more aligned training and validation accuracies.
- **Better Validation Loss**: The validation loss in the optimized model decreased to 0.2923, indicating a more generalized model performance compared to the initial model’s 0.3538 validation loss.


