# CIFAR-10-PROJECT


Project Title: CIFAR-10 Image Classification

Description:
This repository contains a deep learning project for image classification using the CIFAR-10 dataset. The project is implemented in Python using TensorFlow and Keras. It includes modules for loading and displaying the CIFAR-10 dataset, applying data augmentation, building and training a convolutional neural network (CNN) model, and evaluating the model's performance. The code showcases best practices for image classification tasks and serves as a starting point for further experimentation and improvement.

Key Features:

Data loading and preprocessing for CIFAR-10 dataset.
Implementation of a CNN model for image classification.
Application of data augmentation techniques for improved model generalization.
Visualization of training history, confusion matrix, and classification report.
Easily extensible for experimenting with different architectures and hyperparameters.
How to Use:

Clone the repository.
Run main.py to load the dataset, apply data augmentation, and train the CNN model.
Evaluate model performance using accuracy, confusion matrix, and classification report.
Feel free to explore and contribute to enhance the project.

Test Accuracy: 33.37%
Confusion Matrix:
[[329 201  10   3   5   5  17  28 305  97]
 [  4 755   2   1   0   1   5  10  91 131]
 [107 110  68  38 116  56 204 148  47 106]
 [ 45 121  35 119  20 140 160 163  15 182]
 [ 45  64  36  30 175  33 259 245  36  77]
 [ 35  79  22 130  45 199 152 175  32 131]
 [  7  85  51  26  74  37 428 149   6 137]
 [ 19  64  11  33  28  61  46 482  16 240]
 [121 268   8   2   2   1   6   9 500  83]
 [  9 560   3   2   1   7   9  22 105 282]]
Classification Report:
              precision    recall  f1-score   support

    airplane       0.46      0.33      0.38      1000
  automobile       0.33      0.76      0.46      1000
        bird       0.28      0.07      0.11      1000
         cat       0.31      0.12      0.17      1000
        deer       0.38      0.17      0.24      1000
         dog       0.37      0.20      0.26      1000
        frog       0.33      0.43      0.37      1000
       horse       0.34      0.48      0.40      1000
        ship       0.43      0.50      0.46      1000
       truck       0.19      0.28      0.23      1000

    accuracy                           0.33     10000
   macro avg       0.34      0.33      0.31     10000
weighted avg       0.34      0.33      0.31     10000