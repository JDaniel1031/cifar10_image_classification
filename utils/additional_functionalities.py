import tensorflow as tf

def apply_data_augmentation(images):
    # Apply data augmentation using TensorFlow's ImageDataGenerator.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
         rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    datagen.fit(images)
    return datagen
