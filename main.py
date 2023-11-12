import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import logging
from dataset.dataset import load_and_display_dataset
from models.models import build_and_train_complex_model
from utils.additional_functionalities import apply_data_augmentation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint


# Parameterization
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
rotation_range = 20
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.1
zoom_range = 0.1
horizontal_flip = True
fill_mode = 'nearest'
model_checkpoint_path = 'model_weights.h5'

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),  # Save logs to a file
            logging.StreamHandler()  # Display logs on the console
        ]
    )

def load_trained_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def plot_confusion_matrix(conf_matrix, class_names):
    # Display confusion matrix as a heatmap

    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Save the confusion matrix plot as an image
    plt.savefig('confusion_matrix.png')

def cleanup_gpu_memory():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.clear_memory(device)

def main():
    setup_logging()
    logging.info("Starting training and evaluation.")           

    # Load and display the CIFAR-10 dataset.
    train_images, train_labels, test_images, test_labels = load_and_display_dataset()

    # Apply data augmentation.
    augmented_datagen = apply_data_augmentation(train_images)

    # Build and train the model
    model = build_and_train_complex_model(train_images, train_labels, augmented_datagen, test_images, test_labels)

    # Evaluate the model on the test set
    predictions = model.predict(test_images)

    # Convert labels to integers (remove one-hot encoding)
    train_labels_int = np.array(train_labels)
    test_labels_int = np.array(test_labels)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels_int, predictions.argmax(axis=1))
    logging.info(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Create and display confusion matrix
    conf_matrix = confusion_matrix(test_labels_int, predictions.argmax(axis=1))
    logging.info('Confusion Matrix:')
    logging.info(conf_matrix)



    # Display classification report
    class_report = classification_report(test_labels_int, predictions.argmax(axis=1), target_names=class_names)
    logging.info('Classification Report:')
    logging.info(class_report)

    # Save model weights
    model.save_weights(model_checkpoint_path)

    # Save model weights
    model.save_weights(model_checkpoint_path)

    logging.info("Training and evaluation completed.")

    # Run app.py
    #subprocess.Popen(["python", "app.py"])


if __name__ == "__main__":
    main()