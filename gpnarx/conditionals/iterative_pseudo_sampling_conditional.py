from typing import Optional, Tuple, Type, Union, Callable, List

import tensorflow as tf

import tensorflow_probability as tfp

from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.likelihoods import Likelihood
from gpflow.inducing_variables import InducingVariables
from gpflow.conditionals.util import base_conditional_with_lm, sample_mvn
from gpflow import default_jitter, default_float
from gpflow.base import InputData, MeanAndVariance, RegressionData

from gpnarx.utilities import cholesky_rowcolumn_update, cholesky_multiple_rowcolumn_update

class Iterative_pseudo_sampling_conditional_with_precompute:
    '''
        Computes the conditional iteratively and samples from it
    '''
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: MeanFunction,
        likelihood: Likelihood,
        inducing_variable: InducingVariables,
        compile: bool = False
    ):        
        self.data = data
        self.kernel = kernel
        self.mean_function = mean_function
        self.likelihood = likelihood
        self.inducing_variable = inducing_variable
        
        if compile:
            self.predict_f_sample = tf.function(self.predict_f_sample, experimental_relax_shapes=True)
            self.predict_f_sample_init = tf.function(self.predict_f_sample_init)

    def predict_f_sample_init(self, Xnew):
        X_data, Y_data = self.data
        
        L_list, LB_list, A_list, Aerr_list, Z_list, mean_list, var_list = [], [], [], [], [], [], []
        for idx in range(Y_data.shape[-1]):
            likelihood_variance = self.likelihood.likelihoods[idx].variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]
            Z_1d = tf.convert_to_tensor(inducing_variable.Z, dtype = default_float())
            
            X_data_1d, Y_data_1d = X_data, tf.expand_dims(Y_data[..., idx], axis = -1)
            nu = tf.fill(dims = [1, X_data.shape[0]], value = likelihood_variance)
            num_inducing = inducing_variable.num_inducing
            
            assert mean_function is not None
            err = Y_data_1d - mean_function(X_data_1d)
            kuf = kernel(Z_1d, X_data_1d)
            kuu = kernel(Z_1d, full_cov = True)
            
            jitter = default_jitter() * tf.eye(kuu.shape[0], dtype=default_float())
            L_1d = tf.linalg.cholesky(kuu + jitter)  # cache alpha, qinv
            A_1d = tf.linalg.triangular_solve(L_1d, kuf, lower=True)
            B_1d = tf.linalg.matmul(A_1d / tf.sqrt(nu), A_1d / tf.sqrt(nu), transpose_b=True) + tf.eye(
                num_inducing, dtype=default_float()
            )  # cache qinv
            jitter = default_jitter() * tf.eye(B_1d.shape[0], dtype=default_float())
            LB_1d = tf.linalg.cholesky(B_1d + jitter)  # cache alpha     
            
            err = Y_data_1d - mean_function(X_data_1d)
            Aerr_1d = tf.linalg.matmul(A_1d / nu, err)
            
            Kus = kernel(Z_1d, Xnew)
            c = tf.linalg.triangular_solve(LB_1d, Aerr_1d, lower=True)
            tmp1 = tf.linalg.triangular_solve(L_1d, Kus, lower=True)
            tmp2 = tf.linalg.triangular_solve(LB_1d, tmp1, lower=True)
            mean = tf.linalg.matmul(tmp2, c, transpose_a=True)

            var = (
                kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, 1])
            L_list.append(L_1d[..., None])
            LB_list.append(LB_1d[..., None])
            A_list.append(A_1d[..., None])
            Aerr_list.append(Aerr_1d[..., None])
            Z_list.append(Z_1d[..., None])
            mean_list.append(mean), var_list.append(var)

        mean = tf.concat(mean_list, axis = -1)
        var = tf.concat(var_list, axis = -1)
        L = tf.concat(L_list, axis = -1)
        LB = tf.concat(LB_list, axis = -1)
        A = tf.concat(A_list, axis = -1)
        Aerr = tf.concat(Aerr_list, axis = -1)
        Z = tf.concat(Z_list, axis = -1)
        
        # takes a latent sample from the prediction at the next step
        latent_sample = tfp.distributions.Normal(
            loc = tf.squeeze(mean, axis = 0), scale = (tf.squeeze(var, axis = [0, 1]))**0.5
        ).sample()
        latent_sample = tf.expand_dims(latent_sample, axis = 0)
        
        # updates cache with the new latent samples
        X = tf.concat([X_data, Xnew], axis = 0)
        f = tf.concat([Y_data, latent_sample], axis = 0)
        cache = (L, LB, A, Aerr, Z, X, f)
        
        # returns the result
        return latent_sample, cache
        
    def predict_f_sample(
        self,
        Xnew,
        cache: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], #(L, LB, A, Aerr, Z, X, f)
    ):
        (L, LB, A, Aerr, Z, X, f) = cache
    
        N, M = tf.shape(X)[0], self.data[0].shape[0]
        
        LB_list, A_list, Aerr_list, mean_list, var_list  = [], [], [], [], []
        for idx in range(f.shape[-1]):
            likelihood_variance = self.likelihood.likelihoods[idx].variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]     
            Z_1d = tf.convert_to_tensor(inducing_variable.Z, dtype = default_float())
            X_data_1d, Y_data_1d = X, tf.expand_dims(f[..., idx], axis = -1)
            L_1d, LB_1d, A_1d, Aerr_1d = L[..., idx], LB[..., idx], A[..., idx], Aerr[..., idx]
 
            # fill with jitter
            fill_value = tf.constant(default_jitter(), dtype = default_float())
            
            # construct heteroskedastic noise vector
            nu = tf.concat([
                tf.fill(dims = [1, M], value = likelihood_variance), 
                tf.fill(dims = [1, N-M], value = fill_value)
            ], axis = 1)
              
            # Add test point to the inducing variables
            Z_iter = tf.concat([
                Z_1d,
                Xnew
            ], axis = 0)
            num_inducing = inducing_variable.num_inducing + 1
            
            err = Y_data_1d - mean_function(X_data_1d)
            kufnew = kernel(Z_1d, X_data_1d[-1:]) 
            kunewfnew = kernel(Xnew, X_data_1d)
            
            # updates the Kuu Cholesky decomposition
            L_iter_1d = cholesky_multiple_rowcolumn_update(
                X = Z_1d,
                Xnew = Xnew,
                kernel = kernel,
                L = L_1d
            )
            # updates the Kuu Cholesky decomposition
            
            # updates the A solve
            alpha_1d = tf.linalg.triangular_solve(L_1d, kufnew, lower=True) # u x 1
            A_1d = tf.concat([A_1d, alpha_1d], axis = 1) # u x (f + 1)
            beta_1d = (kunewfnew - tf.matmul(L_iter_1d[-1:, :-1], A_1d)) / L_iter_1d[-1, -1] # 1 x (f + 1)
            A_1d_iter = tf.concat([A_1d, beta_1d], axis = 0) # (u + 1) x (f + 1)
            # updates the A solve

            # updates the B Cholesky decomposition
            LB_1d = tfp.math.cholesky_update(
                chol = LB_1d, 
                update_vector = tf.transpose(alpha_1d / tf.sqrt(nu[-1, -1])), 
                multiplier=1.0
            )[0, ...]

            Kcol_update = tf.matmul(A_1d / tf.sqrt(nu), beta_1d / tf.sqrt(nu), transpose_b = True)
            Kcol_update = tf.concat([Kcol_update, (tf.reduce_sum(tf.square(beta_1d) / nu) + tf.constant(1., dtype = default_float()))[None, None]], axis = 0)
            LB_iter_1d = cholesky_rowcolumn_update(
                Kcol_update = Kcol_update,
                L = LB_1d
            )
            # updates the B Cholesky decomposition
            err = Y_data_1d - mean_function(X_data_1d)
            
            # updates Aerr
            Aerr_1d = Aerr_1d + alpha_1d*err[-1, -1]/nu[-1, -1]
            Aerr_iter_1d = tf.concat([Aerr_1d, tf.matmul(beta_1d/nu, err)], axis = 0)
            
            Kus = kernel(Z_iter, Xnew)
            c = tf.linalg.triangular_solve(LB_iter_1d, Aerr_iter_1d, lower=True)
            tmp1 = tf.linalg.triangular_solve(L_iter_1d, Kus, lower=True)
            tmp2 = tf.linalg.triangular_solve(LB_iter_1d, tmp1, lower=True)
            mean = tf.linalg.matmul(tmp2, c, transpose_a=True)

            var = (
                kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, 1])
            
            LB_list.append(LB_1d[..., None])
            A_list.append(A_1d[..., None])
            Aerr_list.append(Aerr_1d[..., None])
            mean_list.append(mean)
            var_list.append(var)

        mean = tf.concat(mean_list, axis = -1)
        mean + self.mean_function(Xnew)
        var = tf.concat(var_list, axis = -1)
        LB = tf.concat(LB_list, axis = -1)
        A = tf.concat(A_list, axis = -1)
        Aerr = tf.concat(Aerr_list, axis = -1)   
        
        # takes a latent sample from the prediction at the next step
        latent_sample = tfp.distributions.Normal(
            loc = tf.squeeze(mean, axis = 0), scale = (tf.squeeze(var, axis = [0, 1]))**0.5
        ).sample()
        latent_sample = tf.expand_dims(latent_sample, axis = 0)
        
        # updates cache with the new latent samples
        X = tf.concat([X, Xnew], axis = 0)
        f = tf.concat([f, latent_sample], axis = 0)
        cache = (L, LB, A, Aerr, Z, X, f)
        
        # returns the result
        return latent_sample, cache
    
