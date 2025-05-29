from snn.models import pavlovian_conditioning

if __name__ == "__main__":
    sim = pavlovian_conditioning(run_time=10.0, D=3, learning_rate=1e-3, plot=True) 