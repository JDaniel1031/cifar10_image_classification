import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.dataset import load_and_display_dataset
from models.models import build_and_train_model
from utils.additional_functionalities import apply_data_augmentation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and display the CIFAR-10 dataset.
train_images, train_labels, test_images, test_labels = load_and_display_dataset()

# Apply data augmentation
augmented_datagen = apply_data_augmentation(train_images)

# Build and train the model
model = build_and_train_model(train_images, train_labels, augmented_datagen, test_images, test_labels)

# Evaluate the model on the test set
predictions = model.predict(test_images)

# Convert labels to integers (remove one-hot encoding)
train_labels_int = np.array(train_labels)
test_labels_int = np.array(test_labels)

# Calculate accuracy
accuracy = accuracy_score(test_labels_int, predictions.argmax(axis=1))
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Create and display confusion matrix
conf_matrix = confusion_matrix(test_labels_int, predictions.argmax(axis=1))
print('Confusion Matrix:')
print(conf_matrix)

# Display confusion matrix as a heatmap
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Display classification report
class_report = classification_report(test_labels_int, predictions.argmax(axis=1), target_names=class_names)
print('Classification Report:')
print(class_report)
