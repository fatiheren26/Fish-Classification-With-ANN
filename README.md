Fish Classification Using Deep Learning

This project implements a deep learning model to classify different species of fish from images. The model is trained to recognize 9 different types of fish using a convolutional neural network (CNN) architecture. The dataset consists of fish images collected from the "A Large-Scale Fish Dataset" which includes various species commonly found in fisheries.
Dataset Description
The dataset includes images of 9 different fish species:

Black Sea Sprat
Gilt-Head Bream
Horse Mackerel
Red Mullet
Red Sea Bream
Sea Bass
Shrimp
Striped Red Mullet
Trout

Each class contains approximately 1000 images, making it a balanced dataset with a total of 9000 images.
Implementation Steps

Data Preparation
Imported necessary libraries including TensorFlow, NumPy, Pandas, and Matplotlib
Created a data frame containing image paths and corresponding labels
Split the dataset into training (70%), validation (30% of training), and test (20%) sets
Implemented data preprocessing using ImageDataGenerator for image augmentation and standardization

Model Architecture
Implemented a Convolutional Neural Network with the following layers:
First convolutional layer: 32 filters, 3x3 kernel size
MaxPooling layer: 2x2 pool size
Second convolutional layer: 64 filters, 3x3 kernel size
MaxPooling layer: 2x2 pool size
Flatten layer
Dense layer with 128 neurons and ReLU activation
Output layer with 9 neurons (for 9 classes) and softmax activation

Training Process
Compiled the model using:

Optimizer: Adam
Loss function: Categorical Crossentropy
Metric: Accuracy

Trained the model for 10 epochs
Implemented early stopping to prevent overfitting

Results and Evaluation
The model achieved impressive results:
Training accuracy: 100%
Validation accuracy: 93.66%
Test accuracy: 94%

Performance metrics per class:

Most classes achieved over 90% precision and recall
Shrimp class showed the highest precision (100%)
Trout class showed slightly lower performance (87% F1-score)

Conclusion
The project successfully demonstrates the application of deep learning for fish species classification, achieving high accuracy across different species. The model shows robust performance in distinguishing between similar-looking fish species, making it potentially valuable for practical applications in fisheries and marine biology research.

Kaggle link: https://www.kaggle.com/code/fatiherenn/fish-classification-with-ann  
