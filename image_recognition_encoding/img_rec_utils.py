import numpy as np
import tensorflow as tf
from nengo_extras.data import one_hot_from_labels
from nengo_extras.matplotlib import tile

def load_and_preprocess_img_rec(normalize_range=(-1, 1)):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    
    X_train = normalize_range[0] + (X_train / 255.0) * (normalize_range[1] - normalize_range[0])
    X_test = normalize_range[0] + (X_test / 255.0) * (normalize_range[1] - normalize_range[0])
    
    T_train = one_hot_from_labels(y_train, classes=10)
    
    return X_train, y_train, T_train, X_test, y_test

def get_outs(simulator, ensemble, conn, images):
    _, acts = nengo.utils.ensemble.tuning_curves(ensemble, simulator, inputs=images)
    return np.dot(acts, simulator.data[conn].weights.T)

def get_error(simulator, ensemble, conn, images, labels):
    return np.argmax(get_outs(simulator, ensemble, conn, images), axis=1) != labels

def print_error(simulator, ensemble, conn, X_train, y_train, X_test, y_test):
    train_error = 100 * get_error(simulator, ensemble, conn, X_train, y_train).mean()
    test_error = 100 * get_error(simulator, ensemble, conn, X_test, y_test).mean()
    print(f"Train/test error: {train_error:.2f}%, {test_error:.2f}%")
    return train_error, test_error

def plot_encoders(encoders, img_shape=(28, 28), rows=4, cols=6):
    tile(encoders.reshape((-1, img_shape[0], img_shape[1])), rows=rows, cols=cols, grid=True) 