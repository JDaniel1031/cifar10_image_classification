import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

def load_and_display_dataset():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Display or preprocess the dataset as needed
    # ...

    return train_images, train_labels, test_images, test_labels
