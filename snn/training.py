import numpy as np
import tensorflow as tf
import nengo
import nengo_dl
import matplotlib.pyplot as plt
from .datasets import load_dataset, load_and_preprocess_mnist, prepare_data
from .plotting import plot_predictions, plot_confusion_matrix_from_preds, print_classification_report, plot_mnist_examples, plot_results
from .models import build_network, train_and_evaluate, evaluate_loihi

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

def run_snn_model(
    input_shape=(1, 28, 28),
    n_parallel=2,
    dt=0.001,
    presentation_time=0.1,
    max_rate=100,
    minibatch_size=200,
    epochs=5,
    do_training=False,
    n_presentations=50,
    n_plots=3,
    param_file="mnist_params.npz",
    folder_path="params",
    snip_max_spikes_per_step=120,
    seed=0
):
    """Run the complete SNN model training and evaluation pipeline."""
    train_images, train_labels, test_images, test_labels = load_and_preprocess_mnist()
    plot_mnist_examples(train_images, train_labels, n_examples=n_plots)
    
    train_images, train_labels, test_images, test_labels = prepare_data(
        train_images, train_labels, test_images, test_labels, presentation_time, dt, minibatch_size
    )
    
    net, inp, out, out_p, out_p_filt = build_network(
        input_shape, n_parallel, dt, presentation_time, max_rate, seed
    )
    
    # Train or load parameters
    sim = train_and_evaluate(
        net, inp, out_p_filt, train_images, train_labels, test_images, test_labels,
        minibatch_size, epochs, do_training, param_file, folder_path
    )
    
    # Add synapse to connections
    for conn in net.all_connections:
        conn.synapse = 0.005
    
    # Evaluate with synapse
    if do_training:
        with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim_eval:
            sim_eval.compile(loss={out_p_filt: classification_accuracy})
            print(
                f"Accuracy w/ synapse: {sim_eval.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)['loss']:.2f}%"
            )
    
    # Evaluate on Loihi
    loihi_accuracy = evaluate_loihi(
        net, out_p_filt, test_images, test_labels, presentation_time, dt,
        n_presentations, snip_max_spikes_per_step
    )
    
    # Plot results
    plot_results(test_images, test_labels, sim, out_p_filt, presentation_time, dt, n_plots) 