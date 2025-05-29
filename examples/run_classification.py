from snn.training import classification

if __name__ == "__main__":
    sim, data = classification(
        do_training=True,
        dataset_name='fashion_mnist',
        epochs=30,
        minibatch_size=100,
        plot=True
    ) 