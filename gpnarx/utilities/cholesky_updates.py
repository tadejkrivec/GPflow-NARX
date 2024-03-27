import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Kernel
from gpflow.config import default_float, default_jitter

def cholesky_rowcolumn_update(
    Kcol_update: tf.Tensor, # new columns of the Gramm matrix
    L: tf.Tensor, # Cached Cholesky decomposition
    observed_latent: bool = True,
    likelihood_variance: tf.Tensor = None
):
    """
        Computes Iterative cholesky updates in O(n^2) instead of O(n^3)
        # https://www5.in.tum.de/pub/sieler16.pdf, page 18
        
        Handles the addition of 1 row/column to the Gramm matrix + n rank-1 updates
    """
    # get the cross covariance terms, and covariance term
    Kmn, Knn = Kcol_update[:-1], Kcol_update[-1:]
    Knn += default_jitter()
    
    if observed_latent == False:
        Knn += likelihood_variance
    
    # get the shape of the process
    M = tf.shape(Kmn)[0]
    
    # iteratively update Cholesky
    c = tf.linalg.triangular_solve(L, Kmn, lower=True) # [M, 1]
    alpha = tf.reduce_sum(tf.square(c), axis = 0) #[1,]
    d = tf.math.sqrt(tf.transpose(Knn) - alpha) #[1, 1] 
    col_update = tf.concat(
        [tf.zeros(shape = (M, 1), dtype = d.dtype), d], axis = 0
    ) # [M + 1, 1]
    row_update = tf.transpose(c, perm = [1, 0]) # [1, M]
    L = tf.concat([L, row_update], axis = 0) # [M + 1, M]
    L = tf.concat([L, col_update], axis = 1) # [M + 1, M + 1]
    return L

def cholesky_rankn_update(
    beta: tf.Tensor, # n x rank-1 updates to Gramm matrix: Knew = K + beta beta^T
    L: tf.Tensor
):
    """
        Handles n rank-1 updates to Gramm matrix: Knew = K + beta beta^T
    """
    # update cholesky for rank n updates to Gramm matrix
    c = lambda i, L: tf.less(i, tf.shape(beta)[1])
    def b(i, L):
        x = beta[..., i][None, ...]
        L = tfp.math.cholesky_update(
            chol = L, 
            update_vector = x, 
            multiplier=1.0
        )[0, ...]
        i = i + 1
        return [i, L]
    
    # initialize the loop variables
    i0 = tf.constant(0)
    
    # run the while loop
    i, L = tf.while_loop(
        cond = c,
        body = b,
        loop_vars = [i0, L],
        shape_invariants = [
            i0.get_shape(), # i
            L.get_shape() # L
        ]
    )
        
    return L

def cholesky_multiple_rowcolumn_update(
    X: tf.Tensor,
    Xnew: tf.Tensor,
    kernel: Kernel,
    L: tf.Tensor, # Cached Cholesky decomposition
    observed_latent: bool = True,
    likelihood_variance: tf.Tensor = None
):
    # Sequentially update the Cholesky decomposition for new data
    # cunstruct functions to be run in tf while loop
    c = lambda i, X, L: tf.less(i, tf.shape(Xnew)[0])
    def b(i, X, L):
        x = Xnew[i][None, ...]
        X = tf.concat([X, x], axis = 0)
        Kcol_update = kernel(
            X, x
        )
        L = cholesky_rowcolumn_update(
            Kcol_update = Kcol_update,
            L = L,
            observed_latent = observed_latent,
            likelihood_variance = likelihood_variance
        )
        i = i + 1
        return [i, X, L]
    
    # initialize the loop variables
    i0 = tf.constant(0)
    
    # run the while loop
    i, X, L = tf.while_loop(
        cond = c,
        body = b,
        loop_vars = [i0, X, L],
        shape_invariants = [
            i0.get_shape(), # i
            tf.TensorShape([None, Xnew.shape[1]]), # X 
            tf.TensorShape([None, None]) # L
        ]
    )

    return L

def cholesky_batch_update(
    Kcol_update: tf.Tensor, # new columns of the Gramm matrix
    L: tf.Tensor, # Cached Cholesky decomposition
    observed_latent: bool = True,
    likelihood_variance: tf.Tensor = None
):
    """
        Computes Iterative cholesky updates in O(n^2) instead of O(n^3)
        # https://www5.in.tum.de/pub/sieler16.pdf, page 18
        
        Handles the addition of 1 row/column to the Gramm matrix + n rank-1 updates
    """
    # get the cross covariance terms, and covariance term
    Kmn, Knn = Kcol_update[:, :-1], Kcol_update[:, -1:]
    Knn += default_jitter()
    
    if observed_latent == False:
        Knn += likelihood_variance
    
    # get the shape of the process
    P = tf.shape(Kmn)[0]
    M = tf.shape(Kmn)[1]
    
    # iteratively update Cholesky
    c = tf.linalg.triangular_solve(L, Kmn, lower=True)
    alpha = tf.reduce_sum(tf.square(c), axis = 1)[..., None]
    d = tf.math.sqrt(Knn - alpha)
    col_update = tf.concat(
        [tf.zeros(shape = (P, M, 1), dtype = d.dtype), d], axis = 1
    )
    row_update = tf.transpose(c, perm = [0, 2, 1])
    L = tf.concat([L, row_update], axis = 1)
    L = tf.concat([L, col_update], axis = 2)
    return L

def kernel_batched(A, B, kernel):
    K = tf.TensorArray(dtype = default_float(), size = A.shape[0])
    c = lambda iteration, _: tf.less(iteration, A.shape[0])

    def b(iteration, K):
        K = K.write(iteration, kernel(A[iteration], B[iteration], full_cov = True))
        iteration = iteration + 1
        return iteration, K
    iteration = tf.constant(0, dtype = tf.int32)
    iteration, K = tf.while_loop(c, b, [iteration, K], parallel_iterations = 100)
    K = K.stack()
    return K