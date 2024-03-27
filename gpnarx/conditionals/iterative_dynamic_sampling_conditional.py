from typing import Optional, Tuple, Type, Union, Callable, List

import tensorflow as tf

import tensorflow_probability as tfp

from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.likelihoods import Likelihood
from gpflow.conditionals.util import base_conditional_with_lm, sample_mvn
from gpflow import default_jitter, default_float

from gpnarx.utilities import cholesky_batch_update

class Iterative_dynamic_sampling_conditional_with_precompute:
    '''
        Computes the conditional iteratively and samples from it
    '''
    def __init__(
        self,
        kernel: Kernel,
        mean_function: MeanFunction,
        likelihood: Likelihood,
        shapes: tuple,
        std_threshold = tf.constant(0., dtype = default_float()),
        compile: bool = False
    ):
        self.kernel = kernel
        self.mean_function = mean_function
        self.likelihood = likelihood
        self.std_threshold = std_threshold
        
        (N_SAMPLES, OUTPUT_DIM, INPUT_DIM) = shapes
        
        if compile:
            self.predict_f_sample = tf.function(
                self.predict_f_sample, 
                input_signature = (
                    tf.TensorSpec(shape=[N_SAMPLES,1,INPUT_DIM], dtype=default_float()),
                    tf.TensorSpec(shape=[N_SAMPLES,None,None,OUTPUT_DIM], dtype=default_float()),
                    tf.TensorSpec(shape=[N_SAMPLES,None,INPUT_DIM,OUTPUT_DIM], dtype=default_float()),
                    tf.TensorSpec(shape=[N_SAMPLES,None,OUTPUT_DIM], dtype=default_float()),
                )
            )
        
    def predict_f_sample(
        self,
        Xnew,
        L, 
        X, 
        f
    ):
        '''
            Computes the prediction at the next step and updates the Cholesky
        '''

         # unpacks the cache
        Xnew = tf.repeat(Xnew[..., None], repeats = X.shape[-1], axis = -1)
        
        # iterates over output dimensions
        mean_list, var_list = [], []
        for idx, kernel in enumerate(self.kernel.kernels):
            # unpacks for a single dimension
            mean_function = self.mean_function.mean_functions[idx]
            f_1d = tf.expand_dims(f[..., idx], axis = -1)
            X_1d = X[..., idx]
            Xnew_1d = Xnew[..., idx]
            L_1d = L[..., idx]
            
            Kmn = kernel_batched(X_1d, Xnew_1d, kernel)
            Knn = kernel(Xnew_1d, full_cov=False)

            err = f_1d - mean_function(X_1d)
            A = tf.linalg.triangular_solve(L_1d, Kmn, lower=True)
            
            fvar = Knn - tf.reduce_sum(tf.square(A), -2)
            A = tf.linalg.triangular_solve(tf.linalg.adjoint(L_1d), A, lower=False)
            fmean = tf.linalg.matmul(A, err, transpose_a=True)
            
            # append the results for a single output dimension
            fmean = fmean + mean_function(Xnew_1d)
            mean_list.append(fmean[..., 0]), var_list.append(fvar)

        # joins the results of a single dimensions
        mean = tf.concat(mean_list, axis = -1)
        var = tf.concat(var_list, axis = -1)
        
        # takes a latent sample from the prediction at the next step
        latent_sample = tfp.distributions.Normal(
            loc = mean, scale = var**0.5
        ).sample()
        
        # update cache
        if tf.math.reduce_any(tf.math.sqrt(var) > self.std_threshold) == tf.constant(True):
            N_SAMPLES = tf.shape(L)[0]
            L_list = []
            for idx, kernel in enumerate(self.kernel.kernels):
                X_1d = X[..., idx]
                L_1d = L[..., idx]
                Xnew_1d = Xnew[..., idx]   
                Kcol_update = kernel_batched(
                    tf.concat([X_1d, Xnew_1d], axis = 1), 
                    Xnew_1d, 
                    kernel
                )
                # updates the Cholesky decomposition
                L_1d = cholesky_batch_update(
                    Kcol_update = Kcol_update,
                    L = L_1d
                )
                L_list.append(L_1d[..., None])
                    
            L = tf.concat(L_list, axis = -1)
            X = tf.concat([X, Xnew], axis = 1)
            f = tf.concat([f, latent_sample[:, None, :]], axis = 1)
        
        # returns the result
        return latent_sample, L, X, f
    
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