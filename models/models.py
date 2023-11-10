import tensorflow as tf
from tensorflow.keras import layers, models


def build_and_train_model(train_images, train_labels, augmented_datagen, test_images, test_labels):
    model = tf.keras.Sequential([
        # Add your model layers here
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Use 'softmax' for multiclass classification
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model..
    model.fit(augmented_datagen.flow(train_images, train_labels, batch_size=32),
              epochs=10,
              validation_data=(test_images, test_labels))

    return model
