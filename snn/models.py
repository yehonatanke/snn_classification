import nengo
import numpy as np
import matplotlib.pyplot as plt
from nengo.processes import WhiteSignal

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