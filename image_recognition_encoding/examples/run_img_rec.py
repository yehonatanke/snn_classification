import nengo
from ..network import run_img_rec_encoding_pipeline

def main():
    run_img_rec_encoding_pipeline(
        n_hid=1000,  # Number of hidden neurons
        img_shape=(28, 28),  # image shape
        patch_size=(11, 11),  # Size of sparse patches
        normalize_range=(-1, 1),  # Normalization range
        neuron_type=nengo.LIFRate(),  # Neuron type
        intercepts=0.1,  # Neuron intercepts
        max_rates=100,  # Maximum firing rates
        reg=0.01,  # Regularization parameter
        seed=3,  # Network seed
        rng_seed=9,  # Random number generator seed
        encoder_types=['normal', 'sparse_normal', 'gabor', 'sparse_gabor'],  # Types of encoders to evaluate
        plot_rows=4,  # Number of rows in encoder visualization
        plot_cols=6  # Number of columns in encoder visualization
    )

if __name__ == "__main__":
    main() 