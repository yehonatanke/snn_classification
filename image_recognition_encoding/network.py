import nengo
import numpy as np
from . import img_rec_utils
from . import encoders

def build_network(
    n_vis=784,
    n_out=10,
    n_hid=1000,
    neuron_type=nengo.LIFRate(),
    intercepts=0.1,
    max_rates=100,
    reg=0.01,
    seed=3
):
    ens_params = dict(
        neuron_type=neuron_type,
        intercepts=nengo.dists.Choice([intercepts]),
        max_rates=nengo.dists.Choice([max_rates]),
    )
    
    solver = nengo.solvers.LstsqL2(reg=reg)
    
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(n_hid, n_vis, **ens_params)
        v = nengo.Node(size_in=n_out)
        conn = nengo.Connection(a, v, synapse=None, solver=solver)
    
    return model, a, conn, v

def run_img_rec_encoding_pipeline(
    n_hid=1000,
    img_shape=(28, 28),
    patch_size=(11, 11),
    normalize_range=(-1, 1),
    neuron_type=nengo.LIFRate(),
    intercepts=0.1,
    max_rates=100,
    reg=0.01,
    seed=3,
    rng_seed=9,
    encoder_types=['normal', 'sparse_normal', 'gabor', 'sparse_gabor'],
    plot_rows=4,
    plot_cols=6
):
    rng = np.random.RandomState(rng_seed)
    
    X_train, y_train, T_train, X_test, y_test = img_rec_utils.load_and_preprocess_img_rec(normalize_range)
    
    model, a, conn, v = build_network(
        n_vis=img_shape[0] * img_shape[1],
        n_out=10,
        n_hid=n_hid,
        neuron_type=neuron_type,
        intercepts=intercepts,
        max_rates=max_rates,
        reg=reg,
        seed=seed
    )
    
    conn.eval_points = X_train
    conn.function = T_train
    
    for encoder_type in encoder_types:
        print(f"\nEvaluating {encoder_type} encoders:")
        
        encoders = encoders.set_encoders(a, encoder_type, n_hid, img_shape, patch_size, rng)
        
        img_rec_utils.plot_encoders(encoders, img_shape, plot_rows, plot_cols)
        
        with nengo.Simulator(model) as sim:
            img_rec_utils.print_error(sim, a, conn, X_train, y_train, X_test, y_test) 