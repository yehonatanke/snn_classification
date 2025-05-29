import nengo
import numpy as np
import matplotlib.pyplot as plt
from nengo.processes import WhiteSignal
import nengo_dl
import nengo_loihi
import tensorflow as tf
import os

# Communication channel
def communication_channel(run_time=10.0, n_neurons=60, white_signal_high=5, white_signal_rms=0.5, seed=None, plot=True):
    """PES learning for a communication channel."""
    model = nengo.Network('Learn a Communication Channel', seed=seed)
    with model:
        stim = nengo.Node(output=WhiteSignal(run_time, high=white_signal_high, rms=white_signal_rms, seed=seed))
        pre = nengo.Ensemble(n_neurons, dimensions=1)
        post = nengo.Ensemble(n_neurons, dimensions=1)
        nengo.Connection(stim, pre)
        conn = nengo.Connection(pre, post, function=lambda x: np.random.random())
        inp_p = nengo.Probe(stim)
        pre_p = nengo.Probe(pre, synapse=0.01)
        post_p = nengo.Probe(post, synapse=0.01)
        error = nengo.Ensemble(n_neurons, dimensions=1)
        error_p = nengo.Probe(error, synapse=0.03)
        nengo.Connection(post, error)
        nengo.Connection(pre, error, transform=-1)
        conn.learning_rule_type = nengo.PES()
        nengo.Connection(error, conn.learning_rule)
    sim = nengo.Simulator(model)
    sim.run(run_time)
    if plot:
        t = sim.trange()
        plt.figure(figsize=(12, 4))
        plt.plot(t, sim.data[inp_p].T[0], color='grey', linewidth=1, label='Input')
        plt.plot(t, sim.data[pre_p].T[0], color='blue',linewidth=1, label='Pre')
        plt.plot(t, sim.data[post_p].T[0], color='red', linewidth=1,label='Post')
        plt.ylabel("Value")
        plt.legend(loc='best', fontsize="small")
        plt.figure(figsize=(12, 4))
        plt.plot(t, sim.data[error_p].T[0], color='black', linewidth=1,  label='Error')
        plt.ylabel("Value")
        plt.xlabel("Time (sec)")
        plt.tight_layout()
        plt.legend(loc='best', fontsize="small")
        plt.show()
    return sim

def pavlovian_conditioning(run_time=15.0, D=3, N=None, learning_rate=3e-4, plot=True, seed=None):
    """Classical conditioning."""
    if N is None:
        N = D * 100
    def us_stim(t):
        t = t % 3
        if 0.9 < t < 1: return [1, 0, 0]
        if 1.9 < t < 2: return [0, 1, 0]
        if 2.9 < t < 3: return [0, 0, 1]
        return [0, 0, 0]
    def cs_stim(t):
        t = t % 3
        if 0.7 < t < 1: return [0.7, 0, 0.5]
        if 1.7 < t < 2: return [0.6, 0.7, 0.8]
        if 2.7 < t < 3: return [0, 1, 0]
        return [0, 0, 0]
    def stop_learning(t):
        return 0 if 8 > t > 2 else 1
    model = nengo.Network(label="Classical Conditioning", seed=seed)
    with model:
        us_stim_node = nengo.Node(us_stim)
        us_stim_p = nengo.Probe(us_stim_node)
        us = nengo.Ensemble(N, D)
        ur = nengo.Ensemble(N, D)
        us_p = nengo.Probe(us, synapse=0.1)
        ur_p = nengo.Probe(ur, synapse=0.1)
        nengo.Connection(us, ur)
        nengo.Connection(us_stim_node, us[:D])
        cs_stim_node = nengo.Node(cs_stim)
        cs_stim_p = nengo.Probe(cs_stim_node)
        cs = nengo.Ensemble(N*2, D*2)
        cr = nengo.Ensemble(N, D)
        cs_p = nengo.Probe(cs, synapse=0.1)
        cr_p = nengo.Probe(cr, synapse=0.1)
        nengo.Connection(cs_stim_node, cs[:D])
        nengo.Connection(cs[:D], cs[D:], synapse=0.2)
        learn_conn = nengo.Connection(cs, cr, function=lambda x: [0]*D)
        learn_conn.learning_rule_type = nengo.PES(learning_rate=learning_rate)
        error = nengo.Ensemble(N, D)
        error_p = nengo.Probe(error, synapse=0.01)
        nengo.Connection(error, learn_conn.learning_rule)
        nengo.Connection(ur, error, transform=-1)
        nengo.Connection(cr, error, transform=1, synapse=0.1)
        stop_learn = nengo.Node(stop_learning)
        stop_learn_p = nengo.Probe(stop_learn)
        nengo.Connection(stop_learn, error.neurons, transform=-10*np.ones((N, 1)))
    sim = nengo.Simulator(model)
    sim.run(run_time)
    if plot:
        t = sim.trange()
        plots = [
            (us_stim_p, ur_p, ['US #1', 'US #2', 'US #3'], ['UR #1', 'UR #2', 'UR #3']),
            (cs_stim_p, ur_p, ['CS #1', 'CS #2', 'CS #3'], ['UR #1', 'UR #2', 'UR #3']),
            (cs_stim_p, cr_p, ['CS #1', 'CS #2', 'CS #3'], ['CR #1', 'CR #2', 'CR #3'])
        ]
        colors = ['red', 'blue', 'black']
        for stim_p, resp_p, stim_labels, resp_labels in plots:
            plt.figure(figsize=(12, 4))
            for i in range(3):
                plt.plot(t, sim.data[stim_p].T[i], c=colors[i], label=stim_labels[i])
                plt.plot(t, sim.data[resp_p].T[i], c=colors[i], label=resp_labels[i], linestyle=":", linewidth=3)
            plt.ylabel("Value")
            plt.xlabel("Time (sec)")
            plt.legend()
            plt.tight_layout()
            plt.show()
        plt.figure(figsize=(12, 4))
        plt.plot(t, sim.data[error_p].T[0], c='grey',linewidth=1, label='error')
        plt.ylabel("Value")
        plt.xlabel("Time (sec)")
        plt.legend(loc='best', fontsize="small")
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(12, 2))
        plt.plot(t, sim.data[stop_learn_p].T[0], c='grey',linewidth=1, label='stop')
        plt.ylabel("Value")
        plt.xlabel("Time (sec)")
        plt.legend(loc='best', fontsize="small")
        plt.show()
        plt.tight_layout()
    return sim

def lif_tuning_curve(Rm=1000, Cm=5e-6, t_ref=10e-3, v_th=1, la=0.05, plot=True):
    """Plot differentiable LIF tuning curves."""
    tau = Rm * Cm
    rho = lambda x: np.max(x, 0)
    rho2 = lambda x: la * np.log(1 + np.exp(x / la))
    a3 = lambda i: 1 / (t_ref + tau * np.log(1 + (v_th / rho(i - v_th))))
    a4 = lambda i: 1 / (t_ref + tau * np.log(1 + (v_th / rho2(i - v_th))))
    I = np.linspace(0, 3, 10000)
    A3 = [a3(i) for i in I]
    A4 = [a4(i) for i in I]
    if plot:
        plt.plot(I, A3, color='grey', label=r'$\rho=max(x,0)$')
        plt.plot(I, A4, color='orange', label=r'$\rho=\lambda log(1+e^{x/\lambda})$')
        plt.ylim(0, 100)
        plt.xlim(0, 3)
        plt.legend(loc='best', fontsize="small")
        plt.ylabel("Firing rate (Hz)")
        plt.xlabel("Input current")
        plt.tight_layout()
        plt.show()
    return I, A3, A4

def conv_layer(x, n_filters, input_shape, kernel_size=(1, 1), strides=(1, 1), activation=True, init=np.ones):
    """Create a convolutional layer with optional activation."""
    conv = nengo.Convolution(
        n_filters, input_shape, channels_last=False, kernel_size=kernel_size, strides=strides, init=init
    )
    layer = nengo.Ensemble(conv.output_shape.size, 1).neurons if activation else nengo.Node(size_in=conv.output_shape.size)
    nengo.Connection(x, layer, transform=conv)
    
    print("LAYER")
    print(conv.input_shape.shape, "->", conv.output_shape.shape)
    
    return layer, conv

def build_network(
    input_shape=(1, 28, 28),
    n_parallel=2,
    dt=0.001,
    presentation_time=0.1,
    max_rate=100,
    seed=0
):
    """Build the neural network for MNIST classification."""
    amp = 1 / max_rate
    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)
        net.config[nengo.Ensemble].neuron_type = nengo.SpikingRectifiedLinear(amplitude=amp)
        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        inp = nengo.Node(
            nengo.processes.PresentInput(np.zeros((1, input_shape[1] * input_shape[2])), presentation_time),
            size_out=input_shape[1] * input_shape[2]
        )
        out = nengo.Node(size_in=10)

        for _ in range(n_parallel):
            layer, conv = conv_layer(inp, 1, input_shape, kernel_size=(1, 1), init=np.ones((1, 1, 1, 1)))
            net.config[layer.ensemble].on_chip = False
            layer, conv = conv_layer(layer, 6, conv.output_shape, strides=(2, 2))
            layer, conv = conv_layer(layer, 24, conv.output_shape, strides=(2, 2))
            nengo.Connection(layer, out, transform=nengo_dl.dists.Glorot())

        out_p = nengo.Probe(out, label="out_p")
        out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01), label="out_p_filt")
    
    return net, inp, out, out_p, out_p_filt

def load_file_from_folder(fname, folder_path='params/mnist_params.npz'):
    """Load parameters from a file."""
    file_path = os.path.join(folder_path, fname)
    if os.path.exists(file_path):
        return
    raise RuntimeError(
        f"Cannot find '{fname}' in {folder_path}. "
    )

def train_and_evaluate(
    net, inp, out_p_filt, train_images, train_labels, test_images, test_labels,
    minibatch_size=200, epochs=5, do_training=False, param_file="mnist_params.npz", folder_path="params"
):
    """Train and evaluate the network using Nengo-DL."""
    with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=0) as sim:
        if do_training:
            sim.compile(loss={out_p_filt: classification_accuracy})
            print(
                f"Accuracy before training: {sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)['loss']:.2f}%"
            )
            sim.compile(
                optimizer=tf.optimizers.RMSprop(0.001),
                loss={out_p_filt: tf.losses.SparseCategoricalCrossentropy(from_logits=True)}
            )
            sim.fit(train_images, train_labels, epochs=epochs)
            sim.compile(loss={out_p_filt: classification_accuracy})
            print(
                f"Accuracy after training: {sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)['loss']:.2f}%"
            )
            sim.save_params(os.path.join(folder_path, param_file))
        else:
            load_file_from_folder(param_file, folder_path)
            sim.load_params(os.path.join(folder_path, param_file))
        sim.freeze_params(net)
    return sim

def evaluate_loihi(
    net, out_p_filt, test_images, test_labels, presentation_time, dt,
    n_presentations=50, snip_max_spikes_per_step=120
):
    """Evaluate the network on Loihi hardware."""
    hw_opts = dict(snip_max_spikes_per_step=snip_max_spikes_per_step)
    with nengo_loihi.Simulator(net, dt=dt, precompute=False, hardware_options=hw_opts) as sim:
        sim.run(n_presentations * presentation_time)
        step = int(presentation_time / dt)
        output = sim.data[out_p_filt][step - 1 :: step]
        correct = 100 * np.mean(np.argmax(output, axis=-1) == test_labels[:n_presentations, -1, 0])
        print(f"Loihi accuracy: {correct:.2f}%")
    return correct 