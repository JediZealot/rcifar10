# rcifar10
CIFAR-10 image classification using CNN on R

This R project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into ten distinct categories. We aim to demonstrate the fundamental principles of CNN architecture and training for image recognition. The model is built using TensorFlow and Keras, providing a practical example of deep learning in image classification.

Modern deep learning frameworks offer the advantage of GPU acceleration, significantly reducing training times. While NVIDIA GPUs currently provide the most robust support on Windows, particularly for TensorFlow, this project was executed on a CPU due to AMD GPU compatibility challenges. As a result, the model training required approximately 10 minutes. This highlights the potential performance gains achievable with optimized GPU utilization and emphasizes the ongoing efforts to expand GPU support across diverse hardware platforms.

The model: 

Input (32x32x3) --> [Conv2D (3x3, 32, ReLU)] --> [Conv2D (3x3, 32, ReLU)] --> [MaxPooling2D (2x2)] --> [Dropout (0.25)] -->
[Conv2D (3x3, 64, ReLU)] --> [Conv2D (3x3, 64, ReLU)] --> [MaxPooling2D (2x2)] --> [Dropout (0.25)] -->
[Flatten] --> [Dense (512, ReLU)] --> [Dropout (0.5)] --> [Dense (10, Softmax)] --> Output (10 classes)

Validation (15 epochs):
Accuracy: 0.7705 - Loss: 0.6681

Test
Accuracy: 0.7637  - Loss: 0.6928567 


Demonstration of the model output (code included)

![image](https://github.com/user-attachments/assets/00d9ccc7-e6a8-4b86-967b-8382ed7e19b2)

Training and Validation Loss (Made using tensorboard)

![Rplot01](https://github.com/user-attachments/assets/0300c6db-5d09-4319-be36-4fd41a156ab3)

