import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import tensorflow as tf

from gpflow import covariances, kernels, mean_functions
from gpflow.base import MeanAndVariance, Module, Parameter, RegressionData, TensorType, InputData
from gpflow.conditionals.util import (
    base_conditional,
    base_conditional_with_lm,
    expand_independent_outputs,
    fully_correlated_conditional,
    independent_interdomain_conditional,
    mix_latent_gp,
    separate_independent_conditional_implementation,
)
from gpflow.config import default_float, default_jitter
from gpflow.covariances import Kuf, Kuu
from gpflow.inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.likelihoods import Likelihood
from gpflow.utilities import Dispatcher, add_noise_cov
from gpflow.utilities.ops import eye, leading_transpose
from gpflow.posteriors import _validate_precompute_cache_type, PrecomputedValue, PrecomputeCacheType, _QDistribution, _DeltaDist, _DiagNormal, _MvNormal
from gpflow.conditionals.util import sample_mvn
from gpflow.posteriors import AbstractPosterior 

class MultivariateAbstractPIMCPosterior(Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        X_data: Union[tf.Tensor, InducingVariables],
        cache: Optional[Tuple[tf.Tensor, ...]] = None,
        mean_function: Optional[mean_functions.MeanFunction] = None,
    ) -> None:

        super().__init__()

        self.kernel = kernel
        self.X_data = X_data
        self.cache = cache
        self.mean_function = mean_function
        self._precompute_cache: Optional[PrecomputeCacheType] = None

    def _add_mean_function(self, Xnew: TensorType, mean: TensorType) -> tf.Tensor:
        if self.mean_function is None:
            return mean
        else:
            return mean + self.mean_function(Xnew)

    @abstractmethod
    def _precompute(self) -> Tuple[PrecomputedValue, ...]:
        """
        Precompute a cache.
        The result of this method will later be passed to `_conditional_with_precompute` as the
        `cache` argument.
        """  

    def predict_f(
        self, 
        Xnew: TensorType, 
        full_cov: bool = False, 
        full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function.
        Relies on precomputed alpha and Qinv (see _precompute method)
        """
        if full_output_cov == True:
            raise NotImplementedError(
                "Multioutput covariance not working for now - TODO"
            )
            
        if self.cache is None:
            raise ValueError(
                "Cache has not been precomputed yet. Call update_cache first or use fused_predict_f"
            )     
        
        mean, cov = self._conditional_with_precompute(
            self.cache, 
            Xnew, 
            full_cov=full_cov, 
            full_output_cov=full_output_cov
        )
            
        num_latent_gps = mean.shape[-1]           
        if self.mean_function is not None:
            mean_list = []
            for idx in range(num_latent_gps):
                mean_function = self.mean_function.mean_functions[idx]
                mean_idx = mean[..., idx][..., None] + mean_function(Xnew)
                mean_list.append(mean_idx)
            mean = tf.concat(mean_list, axis = -1)
        else:
            mean = mean   
        var = cov 
        
        # returns mean and var
        return mean, var

    @abstractmethod
    def _conditional_with_precompute(
        self,
        cache: Tuple[tf.Tensor, ...],
        Xnew: TensorType,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function.
        Relies on cached alpha and Qinv.
        """

    def predict_f_samples(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = None,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.
        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, D]
            where N is the number of rows and D is the input dimension of each point.
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples and P is the number of outputs.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.
        Currently, the method does not support `full_output_cov=True` and `full_cov=True`.
        """
        if full_output_cov == True:
            raise NotImplementedError(
                "Multioutput covariance not working for now - TODO"
            )
        
        if full_cov and full_output_cov:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not supported."
            )

        # check below for shape info
        mean, cov = self.predict_f(
            Xnew, 
            full_cov=full_cov, 
            full_output_cov=full_output_cov
        )
           
        if full_cov:
            # mean: [..., N, P]
            # cov: [..., P, N, N]
            mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
            samples = sample_mvn(
                mean_for_sample, cov, full_cov, num_samples=num_samples
            )  # [..., (S), P, N]
            samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        else:
            # mean: [..., N, P]
            # cov: [..., N, P] or [..., N, P, P]
            samples = sample_mvn(
                mean, cov, full_output_cov, num_samples=num_samples
            )  # [..., (S), N, P]
        return samples  # [..., (S), N, P]
    
    def update_cache(self, precompute_cache: Optional[PrecomputeCacheType] = None) -> None:
        """
        Sets the cache depending on the value of `precompute_cache` to a
        `tf.Tensor`, `tf.Variable`, or clears the cache. If `precompute_cache`
        is not given, the setting defaults to the most-recently-used one.
        """
        if precompute_cache is None:
            if self._precompute_cache is None:
                raise ValueError(
                    "You must pass precompute_cache explicitly"
                    " (the cache had not been updated before)."
                )
            precompute_cache = self._precompute_cache
        else:
            self._precompute_cache = precompute_cache

        if precompute_cache is PrecomputeCacheType.NOCACHE:
            self.cache = None

        elif precompute_cache is PrecomputeCacheType.TENSOR:
            self.cache = tuple(c.value for c in self._precompute())

        elif precompute_cache is PrecomputeCacheType.VARIABLE:
            cache = self._precompute()

            if self.cache is not None and all(isinstance(c, tf.Variable) for c in self.cache):
                # re-use existing variables
                for cache_var, c in zip(self.cache, cache):
                    cache_var.assign(c.value)
            else:  # create variables
                shapes = [
                    [None if d else s for d, s in zip(c.axis_dynamic, tf.shape(c.value))]
                    for c in cache
                ]
                self.cache = tuple(
                    tf.Variable(c.value, trainable=False, shape=s) for c, s in zip(cache, shapes)
                )

class MultivariatePIMCPosterior(MultivariateAbstractPIMCPosterior):
    def __init__(
        self,
        kernel: Kernel,
        data: RegressionData,
        mean_function: MeanFunction,
        L: tf.Tensor,
        *,
        precompute_cache: Optional[PrecomputeCacheType],
    ) -> None:
        X, Y = data
        super().__init__(kernel, X, mean_function=mean_function)
        self.Y_data = Y
        self.num_latent_gps = Y.shape[-1]
        self.L = L    

        if precompute_cache is not None:
            self.update_cache(precompute_cache)

    def _precompute(self) -> Tuple[PrecomputedValue, ...]:
        Kmm_plus_s_inv_list = [] 
        for idx in range(self.num_latent_gps):

            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            X_data_1d, Y_data_1d = self.X_data[..., idx], self.Y_data[..., idx]
            
            Lm = self.L[..., idx]
            Kmm_plus_s_inv = tf.linalg.cholesky_solve(
                Lm, tf.eye(tf.shape(self.X_data)[0], dtype=Lm.dtype)
            )

            M = X_data_1d.shape[0]
            M_dynamic = M is None
   
            Kmm_plus_s_inv_list.append(Kmm_plus_s_inv[..., None])
        Kmm_plus_s_inv = tf.concat(Kmm_plus_s_inv_list, axis = -1) 
        return (PrecomputedValue(Kmm_plus_s_inv, (M_dynamic, M_dynamic, False)),)
            
    def _conditional_with_precompute(
        self,
        cache: Tuple[tf.Tensor, ...],
        Xnew: TensorType,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function.
        Relies on cached alpha and Qinv.
        """
        (Qinv,) = cache
        
        mean_list, var_list = [], []
        for idx in range(self.num_latent_gps):

            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            X_data_1d, Y_data_1d = self.X_data[..., idx], self.Y_data[..., idx]
            Qinv_1d = Qinv[..., idx]
            
            Kmn = kernel(X_data_1d, Xnew)
            # compute kernel stuff
            num_func = 1  # R
            N = tf.shape(Kmn)[-1]

            # get the leading dims in Kmn to the front of the tensor Kmn
            K = tf.rank(Kmn)
            perm = tf.concat(
                [
                    tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
                    tf.reshape(0, [1]),  # [M]
                    tf.reshape(K - 1, [1]),
                ],
                0,
            )  # [N]
            Kmn = tf.transpose(Kmn, perm)  # [..., M, N]
            leading_dims = tf.shape(Kmn)[:-2]

            # get the leading dims in Knm to the front of the tensor Knm
            Knm = leading_transpose(Kmn, [..., -1, -2])

            assert mean_function is not None
            Knn = kernel(Xnew, full_cov=full_cov)
            
            err = tf.transpose(Y_data_1d) - mean_function(X_data_1d)
            mean = (Knm @ Qinv_1d)[:, None, :] @ tf.transpose(err)[:, :, None]
            
            # The GPR model only has a single latent GP.
            if full_cov:
                cov = Knn - Knm @ Qinv_1d @ Kmn  # [..., N, N]
                cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
                cov = tf.broadcast_to(tf.expand_dims(cov, -3), cov_shape)  # [..., R, N, N]

            else:
                cov = Knn - tf.einsum("...ij,...ji->...i", Knm @ Qinv_1d, Kmn)  # [..., N]
                cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
                cov = tf.broadcast_to(tf.expand_dims(cov, -2), cov_shape)  # [..., R, N]
                cov = tf.linalg.adjoint(cov)

            mean_list.append(mean[..., 0]), var_list.append(cov)
            
        # concat
        mean = tf.concat(mean_list, axis = -1)
        if full_cov:
            var = tf.concat(var_list, axis = 0)
        else:
            var = tf.concat(var_list, axis = 1)
            
        return mean, var