import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_mnist():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], -1))
    test_images = test_images.reshape((test_images.shape[0], -1))
    return train_images, train_labels, test_images, test_labels

def prepare_data(train_images, train_labels, test_images, test_labels, presentation_time, dt, minibatch_size=200):
    """Prepare training and testing data for simulation."""
    train_images = train_images[:, None, :]
    train_labels = train_labels[:, None, None]
    n_steps = int(presentation_time / dt)
    test_images = np.tile(test_images[:minibatch_size * 2, None, :], (1, n_steps, 1))
    test_labels = np.tile(test_labels[:minibatch_size * 2, None, None], (1, n_steps, 1))
    return train_images, train_labels, test_images, test_labels

def load_dataset(name='mnist', flatten=True, normalize=True, augment=False, custom_data=None):
    """Load and preprocess dataset."""
    if custom_data is not None:
        (train_images, train_labels), (test_images, test_labels) = custom_data
    elif name == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    elif name == 'fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    if normalize:
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
    if flatten:
        train_images = train_images.reshape((train_images.shape[0], -1))
        test_images = test_images.reshape((test_images.shape[0], -1))
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=10,
            horizontal_flip=True
        )
        train_images = train_images.reshape((-1, 28, 28, 1))
        train_images = next(datagen.flow(train_images, batch_size=len(train_images), shuffle=False))
        if flatten:
            train_images = train_images.reshape((train_images.shape[0], -1))
    return (train_images, train_labels), (test_images, test_labels) 