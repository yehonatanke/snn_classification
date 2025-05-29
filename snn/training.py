import numpy as np
import tensorflow as tf
import nengo
import nengo_dl
import matplotlib.pyplot as plt
from .datasets import load_dataset
from .plotting import plot_predictions, plot_confusion_matrix_from_preds, print_classification_report

def classification(
    do_training=False,
    minibatch_size=200,
    n_steps=30,
    dataset_name='mnist',
    epochs=10,
    learning_rate=0.001,
    neuron_type=None,
    save_path=None,
    plot=True,
    seed=0,
    eval_plots=False
):
    """
    Trains and evaluates the snn.
    Args:
        do_training (bool): Whether to train the model.
        minibatch_size (int): Batch size for training/evaluation.
        n_steps (int): Number of time steps for SNN simulation.
        dataset_name (str): 'mnist' or 'fashion_mnist'.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        neuron_type: Nengo neuron type (default: LIF with amplitude=0.01).
        save_path (str): Path to save trained parameters.
        plot (bool): Whether to plot predictions.
        seed (int): Random seed.
        eval_plots (bool): Whether to plot confusion matrix and classification report after evaluation.
    Returns:
        sim: NengoDL Simulator
        data: Prediction data
    """
    (train_images, train_labels), (test_images, test_labels) = load_dataset(dataset_name)
    train_images = train_images[:, None, :]
    train_labels = train_labels[:, None, None]
    test_images = np.tile(test_images[:, None, :], (1, n_steps, 1))
    test_labels = np.tile(test_labels[:, None, None], (1, n_steps, 1))
    if neuron_type is None:
        neuron_type = nengo.LIF(amplitude=0.01)
    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None
        nengo_dl.configure_settings(stateful=False)
        inp = nengo.Node(np.zeros(train_images.shape[-1]))
        x = nengo_dl.Layer(tf.keras.layers.Reshape((28, 28, 1)))(inp)
        x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(x)
        x = nengo_dl.Layer(neuron_type)(x)
        x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(x)
        x = nengo_dl.Layer(neuron_type)(x)
        x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(x)
        x = nengo_dl.Layer(neuron_type)(x)
        out = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(x)
        out_p = nengo.Probe(out, label="out_p")
        out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")
    sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)
    def classification_accuracy(y_true, y_pred):
        return tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])
    sim.compile(loss={out_p_filt: classification_accuracy})
    print("Accuracy before training:", sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"])
    if do_training:
        sim.compile(optimizer=tf.optimizers.RMSprop(learning_rate),
                   loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)})
        sim.fit(train_images, {out_p: train_labels}, epochs=epochs)
        if save_path:
            sim.save_params(save_path)
        sim.compile(loss={out_p_filt: classification_accuracy})
    print("Accuracy after training:", sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"])
    data = sim.predict(test_images[:minibatch_size])
    if plot:
        show_predictions(data, test_images, out_p_filt, minibatch_size)
    if eval_plots:
        y_true = test_labels[:minibatch_size, -1, 0]
        y_pred = np.argmax(data[out_p_filt][:, -1, :], axis=-1)
        class_names = [str(i) for i in range(10)]
        plot_confusion_matrix_from_preds(y_true, y_pred, class_names)
        print_classification_report(y_true, y_pred, class_names)
    return sim, data

def show_predictions(data, test_images, out_p_filt, minibatch_size=5, test_labels=None):
    plot_predictions(data, test_images, out_p_filt, num_samples=minibatch_size) 