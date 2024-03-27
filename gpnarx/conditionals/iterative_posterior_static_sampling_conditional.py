from typing import Optional, Tuple, Type, Union, Callable, List

import tensorflow as tf

import tensorflow_probability as tfp

from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.likelihoods import Likelihood
from gpflow.conditionals.util import base_conditional_with_lm, sample_mvn
from gpflow import default_jitter, default_float

from gpnarx.utilities import cholesky_multiple_rowcolumn_update

class Iterative_posterior_static_sampling_conditional_with_precompute:
    '''
        Computes the conditional iteratively and samples from it
    '''
    def __init__(
        self,
        L,
        Z,
        f,
        mean_qu, 
        var_qu,
        shapes: tuple,
        std_threshold = tf.constant(0., dtype = default_float()),
        compile: bool = False
    ):
        self.L = L
        self.Z = Z
        self.f = f
        self.mean_qu = mean_qu
        self.var_qu = var_qu
        self.std_threshold = std_threshold
        (N_SAMPLES, OUTPUT_DIM, INPUT_DIM) = shapes
        
        if compile:
            self.predict_f_sample = tf.function(
                self.predict_f_sample, 
                input_signature = (
                    tf.TensorSpec(shape=[1,INPUT_DIM], dtype=default_float()),
                    tf.TensorSpec(shape=[mean_qu.shape[0],], dtype=tf.bool),
                    tf.TensorSpec(shape=[], dtype=tf.int32),
                )
            )
    
    def predict_f_sample(
        self,
        Xnew,
        mask,
        index
    ):
         # unpacks the cache
        Xnew = tf.repeat(Xnew[..., None], repeats = self.Z.shape[-1], axis = -1)
        
        # iterates over output dimensions
        mean_list, var_list = [], []
        for idx in range(self.f.shape[-1]):
            # unpacks for a single dimension
            # mean_function = self.mean_function.mean_functions[idx]
            mf_iter =  tf.boolean_mask(self.mean_qu, mask, axis = 0)
            mf_new = self.mean_qu[index, ...][None, :]
            L_iter = tf.boolean_mask(tf.boolean_mask(self.L, mask, axis = 1), mask, axis = 2)
            X_iter = tf.boolean_mask(self.Z, mask, axis = 0)
            f_iter = tf.boolean_mask(self.f, mask, axis = 1)

            mf_1d = tf.expand_dims(mf_iter[..., idx], axis = -1)
            mf_new_1d = tf.expand_dims(mf_new[..., idx], axis = -1)
            f_1d = tf.expand_dims(f_iter[..., idx], axis = -1)
            X_1d = X_iter[..., idx]
            L_1d = L_iter[..., idx]
            Xnew_1d = Xnew[..., idx]
            var_qu_1d = self.var_qu[idx, ...]
            
            # Kmn = kernel(X_1d, Xnew_1d)
            Kmn = tf.boolean_mask(var_qu_1d, mask, axis = 0)[:, index][:, None]
            # Knn = kernel(Xnew_1d, full_cov=True)
            Knn = var_qu_1d[index, index][None, None]
            
            err = f_1d - mf_1d
            A = tf.linalg.triangular_solve(L_1d, Kmn, lower=True)
            fvar = Knn - tf.reduce_sum(tf.square(A), -2)
            A = tf.linalg.triangular_solve(tf.linalg.adjoint(L_1d), A, lower=False)
            fmean = tf.linalg.matmul(A, err, transpose_a=True)
                
            # append the results for a single output dimension
            fmean = fmean[..., 0] + mf_new_1d
            mean_list.append(fmean), var_list.append(fvar)
            
        # joins the results of a single dimensions
        mean = tf.concat(mean_list, axis = -1)
        var = tf.concat(var_list, axis = -1)
        
        # takes a latent sample from the prediction at the next step
        latent_sample = tfp.distributions.Normal(
            loc = mean, scale = var**0.5
        ).sample()
        latent_sample = tf.expand_dims(latent_sample, axis = 2)
        
        # update cache
        if tf.math.reduce_any(tf.math.sqrt(var) > self.std_threshold) == tf.constant(True):
            mask = tf.concat([mask[:index], tf.constant([True]), mask[index + 1:]], axis = 0)
        
        # returns the result
        return mask