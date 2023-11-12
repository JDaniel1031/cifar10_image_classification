import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler




# Define a learning rate scheduler function
def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr  # Keep the initial learning rate for the first 5 epochs
    else:
        return lr * tf.math.exp(-0.1)  # Decay the learning rate exponentially after epoch 5
    

def build_complex_model():
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Additional Convolutional Layers
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))  # Add dropout for regularization
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))  # Add dropout for regularization
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Model compilation and training
    initial_learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_and_train_complex_model(train_images, train_labels, augmented_datagen, test_images, test_labels):
    # Check if a pre-trained model exists
    model_path = os.path.join('models', 'cifar10_model.h5')
    if os.path.exists('models/cifar10_model.h5'):
        # Load the pre-trained model
        model = load_model('models/cifar10_model.h5')
    else:
        # Create and compile a new model if no pre-trained model exists
        model = build_complex_model()

    # Define a LearningRateScheduler callback
    lr_callback = LearningRateScheduler(lr_scheduler)

    # Define a ModelCheckpoint callback to save the model weights during training
    checkpoint_callback = ModelCheckpoint(os.path.join('models', 'cifar10_model_checkpoint.h5'),  # Full path to save the weights
                                           monitor='val_loss',  # Metric to monitor
                                           save_best_only=True,  # Save only the best model
                                           save_weights_only=True,  # Save only weights, not entire model
                                           mode='min',  # Mode for monitoring ('min' for loss, 'max' for accuracy)
                                           verbose=1)  # Verbosity level

    # Continue training the model
    history = model.fit(augmented_datagen.flow(train_images, train_labels, batch_size=24),
                        epochs=1,
                        validation_data=(test_images, test_labels),
                        callbacks=[checkpoint_callback,lr_callback ])

    # Save the entire model, including weights, after training
    model.save(model_path)

    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return model


    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
