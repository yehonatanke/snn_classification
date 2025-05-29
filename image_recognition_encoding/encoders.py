import numpy as np
from nengo_extras.vision import Gabor, Mask

def set_encoders(a, encoder_type, n_hid, img_shape=(28, 28), patch_size=(11, 11), rng=np.random.RandomState(9)):
    if encoder_type == 'normal':
        encoders = rng.normal(size=(n_hid, img_shape[0] * img_shape[1]))
    elif encoder_type == 'sparse_normal':
        encoders = rng.normal(size=(n_hid, patch_size[0], patch_size[1]))
        encoders = Mask(img_shape).populate(encoders, rng=rng, flatten=True)
    elif encoder_type == 'gabor':
        encoders = Gabor().generate(n_hid, img_shape, rng=rng).reshape((n_hid, -1))
    elif encoder_type == 'sparse_gabor':
        encoders = Gabor().generate(n_hid, patch_size, rng=rng)
        encoders = Mask(img_shape).populate(encoders, rng=rng, flatten=True)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    a.encoders = encoders
    return encoders 