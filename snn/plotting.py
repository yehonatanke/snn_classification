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