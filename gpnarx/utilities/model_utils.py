import tensorflow as tf

from gpflow.base import TensorType

def add_noise_cov_partial(K: tf.Tensor, likelihood_variance: TensorType, p: int) -> tf.Tensor:
    """
    Returns K + [σ² I 0; 0 0], where σ² is the likelihood noise variance (scalar),
    and I € R^(p x p) is the corresponding identity matrix and K € R^(n x n).
    """
    k_diag = tf.linalg.diag_part(K)
    m = tf.shape(k_diag)[0] - p
    s_diag = tf.concat([
        tf.fill((p,), likelihood_variance),
        tf.fill((m,), tf.constant(0., dtype = k_diag.dtype)) 
    ], axis = 0)
    return tf.linalg.set_diag(K, k_diag + s_diag)