import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

# Plot predictions
def plot_predictions(data, test_images, out_p_filt, num_samples=5):
    """Plot predictions for samples."""
    for i in range(num_samples):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(test_images[i, 0].reshape((28, 28)), cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.plot(tf.nn.softmax(data[out_p_filt][i]))
        plt.legend([str(i) for i in range(10)], loc="upper left")
        plt.xlabel("timesteps")
        plt.ylabel("probability")
        plt.tight_layout()

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_from_preds(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """Print classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names))

def spike_raster_plot(spikes, title="Spike Raster"):
    """Plot spike raster."""
    plt.figure(figsize=(10, 4))
    plt.eventplot([np.where(s > 0)[0] for s in spikes], colors='black')
    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Neuron')
    plt.tight_layout()
    plt.show()

def plot_mnist_examples(images, labels, n_examples=3):
    """Plot MNIST example images with their labels."""
    for i in range(n_examples):
        plt.figure()
        plt.imshow(np.reshape(images[i], (28, 28)), cmap="gray")
        plt.axis("off")
        plt.title(str(labels[i]))
        plt.show()

def plot_results(test_images, test_labels, sim, out_p_filt, presentation_time, dt, n_plots=3):
    """Plot input images and network output."""
    step = int(presentation_time / dt)
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    images = test_images.reshape(-1, 28, 28, 1)[::step]
    ni, nj, nc = images[0].shape
    allimage = np.zeros((ni, nj * n_plots, nc), dtype=images.dtype)
    for i, image in enumerate(images[:n_plots]):
        allimage[:, i * nj : (i + 1) * nj] = image
    if allimage.shape[-1] == 1:
        allimage = allimage[:, :, 0]
    plt.imshow(allimage, aspect="auto", interpolation="none", cmap="gray")
    plt.axis("off")
    
    plt.subplot(2, 1, 2)
    plt.plot(sim.trange()[:n_plots * step], sim.data[out_p_filt][:n_plots * step])
    plt.legend([str(i) for i in range(10)], loc="best")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.show() 