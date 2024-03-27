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

class MultivariateAbstractPosterior(Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        X_data: Union[tf.Tensor, InducingVariables],
        cache: Optional[Tuple[tf.Tensor, ...]] = None,
        mean_function: Optional[mean_functions.MeanFunction] = None,
    ) -> None:
        """
        Users should use `create_posterior` to create instances of concrete
        subclasses of this AbstractPosterior class instead of calling this
        constructor directly. For `create_posterior` to be able to correctly
        instantiate subclasses, developers need to ensure their subclasses
        don't change the constructor signature.

        Code based on gpflow.posteriors.AbstractPosterior @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
        
        Added multivariate functionality and the ability to predict with already observed latent data
        """
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
            
    def fused_predict_f(
        self, 
        Xnew: TensorType, 
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        full_cov: bool = False, 
        full_output_cov: bool = False
    ) -> MeanAndVariance:
        
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        if full_output_cov == True:
            raise NotImplementedError(
                "Multioutput covariance not working for now - TODO"
            )
            
        if observed_data is not None:
            X_data_obs, Y_data_obs = observed_data
            M = X_data_obs.shape[0]
            Xnew = tf.concat([Xnew, X_data_obs], axis = 0)
            
        mean, cov = self._conditional_fused(
            Xnew, 
            full_cov=True, 
            full_output_cov=full_output_cov
        )
        mean, var = self._add_mean_function(Xnew, mean), cov
        num_latent_gps = mean.shape[-1]
        
        # condition if observed data exists
        if observed_data is not None:
            mean_list, var_list = [], []
            for idx in range(num_latent_gps):
                likelihood = self.likelihood.likelihoods[idx]
                Y_data_1d = tf.expand_dims(Y_data_obs[..., idx], axis = 1)
                jitter = default_jitter() * tf.eye(M, dtype=default_float())
                
                if observed_latent:
                    L_1d = tf.linalg.cholesky(var[idx, -M:, -M:] + jitter)
                else:
                    lik_var = likelihood.variance * tf.eye(M, dtype=default_float())
                    L_1d = tf.linalg.cholesky(var[idx, -M:, -M:] + jitter + lik_var)

                Kmn = var[idx, -M:, :-M]
                Knn = var[idx, :-M, :-M]
                    
                err = Y_data_1d - tf.expand_dims(mean[-M:, idx], axis = 1)
                mean_n, var_n = base_conditional_with_lm(
                    Kmn, L_1d, Knn, err, full_cov=True, white=False
                )  # [N, P], [N, P] or [P, N, N]
                mean_n += tf.expand_dims(mean[:-M, idx], axis = 1)
                mean_list.append(mean_n), var_list.append(var_n)

            mean = tf.concat(mean_list, axis = -1)
            var = tf.concat(var_list, axis = 0)

        if not full_cov:
            var = tf.linalg.diag_part(var)
        
        return mean, var 

    @abstractmethod
    def _conditional_fused(
        self, 
        Xnew: TensorType, 
        full_cov: bool = False, 
        full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function
        Does not make use of caching
        """

    def predict_f(
        self, 
        Xnew: TensorType, 
        observed_data: RegressionData = None,
        observed_latent: bool = True,
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
            
        if observed_data is not None:
            X_data_obs, Y_data_obs = observed_data
            M = X_data_obs.shape[0]
            Xnew = tf.concat([Xnew, X_data_obs], axis = 0)
        
        if observed_data is not None:
            mean, cov = self._conditional_with_precompute(
                self.cache, 
                Xnew, 
                full_cov=True, 
                full_output_cov=full_output_cov
            )
        else:
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
        
        ### The part that conditions on observed data ####
        # condition if observed data exists
        if observed_data is not None:
            mean_list, var_list = [], []
            for idx in range(num_latent_gps):
                likelihood = self.likelihood.likelihoods[idx]
                Y_data_1d = tf.expand_dims(Y_data_obs[..., idx], axis = 1)
                jitter = default_jitter() * tf.eye(M, dtype=default_float())
                
                if observed_latent:
                    L_1d = tf.linalg.cholesky(var[idx, -M:, -M:] + jitter)
                else:
                    lik_var = likelihood.variance * tf.eye(M, dtype=default_float())
                    L_1d = tf.linalg.cholesky(var[idx, -M:, -M:] + jitter + lik_var)

                Kmn = var[idx, -M:, :-M]
                Knn = var[idx, :-M, :-M]
                
                err = Y_data_1d - tf.expand_dims(mean[-M:, idx], axis = 1)
                mean_n, var_n = base_conditional_with_lm(
                    Kmn, L_1d, Knn, err, full_cov=True, white=False
                )  # [N, P], [N, P] or [P, N, N]
                mean_n += tf.expand_dims(mean[:-M, idx], axis = 1)
                mean_list.append(mean_n), var_list.append(var_n)

            mean = tf.concat(mean_list, axis = -1)
            var = tf.concat(var_list, axis = 0)

            if not full_cov:
                var = tf.linalg.diag_part(var)
                var = tf.transpose(var)
        ### The part that conditions on observed data ####
        
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
        observed_data: RegressionData = None,
        observed_latent: bool = True,
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
            observed_data=observed_data,
            observed_latent=observed_latent,
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

    def fused_predict_f_samples(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = None,
        observed_data: RegressionData = None,
        observed_latent: bool = True,
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
        mean, cov = self.fused_predict_f(
            Xnew, 
            observed_data=observed_data,
            observed_latent=observed_latent,
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

class MultivariateGPRPosterior(MultivariateAbstractPosterior):
    """
        Code based on gpflow.posteriors.GPRPosterior @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
        
        Added multivariate functionality
    """
    def __init__(
        self,
        kernel: Kernel,
        data: RegressionData,
        likelihood: Likelihood,
        mean_function: MeanFunction,
        *,
        precompute_cache: Optional[PrecomputeCacheType],
    ) -> None:
        X, Y = data
        super().__init__(kernel, X, mean_function=mean_function)
        self.Y_data = Y
        self.num_latent_gps = Y.shape[-1]
        self.likelihood = likelihood

        if precompute_cache is not None:
            self.update_cache(precompute_cache)

    def _precompute(self) -> Tuple[PrecomputedValue, ...]:
        Kmm_plus_s_inv_list = [] 
        for idx in range(self.num_latent_gps):
            likelihood = self.likelihood.likelihoods[idx]
            likelihood_variance = likelihood.variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            X_data_1d, Y_data_1d = self.X_data, tf.expand_dims(self.Y_data[..., idx], axis = -1)
            
            Kmm = kernel(X_data_1d)
            Kmm_plus_s = add_noise_cov(Kmm, likelihood_variance)

            Lm = tf.linalg.cholesky(Kmm_plus_s)
            Kmm_plus_s_inv = tf.linalg.cholesky_solve(
                Lm, tf.eye(tf.shape(self.X_data)[0], dtype=Lm.dtype)
            )

            M = X_data_1d.shape[0]
            M_dynamic = M is None
  
            tf.debugging.assert_shapes(
                [
                    (Kmm_plus_s_inv, ["M", "M"]),
                    (Kmm, ["M", "M"]),
                ]
            )    
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
        (alpha,) = cache
        (Qinv,) = cache
        
        mean_list, var_list = [], []
        for idx in range(self.num_latent_gps):
            likelihood = self.likelihood.likelihoods[idx]
            likelihood_variance = likelihood.variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            X_data_1d, Y_data_1d = self.X_data, tf.expand_dims(self.Y_data[..., idx], axis = -1)
            alpha_1d = alpha[..., idx]
            Qinv_1d = Qinv[..., idx]
            
            Kmn = kernel(X_data_1d, Xnew)
            # compute kernel stuff
            num_func = tf.shape(Y_data_1d)[-1]  # R
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
            err = Y_data_1d - mean_function(X_data_1d)

            mean = Knm @ alpha_1d @ err

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

            mean_list.append(mean), var_list.append(cov)
            
        # concat
        mean = tf.concat(mean_list, axis = -1)
        if full_cov:
            var = tf.concat(var_list, axis = 0)
        else:
            var = tf.concat(var_list, axis = 1)
        return mean, var
    
    def _conditional_fused(
        self,
        Xnew: TensorType, 
        full_cov: bool = False, 
        full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function
        Does not make use of caching
        """
        mean_list, var_list = [], []
        for idx in range(self.num_latent_gps): 
            likelihood_variance = self.likelihood.likelihoods[idx].variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            X_data_1d, Y_data_1d = self.X_data, tf.expand_dims(self.Y_data[..., idx], axis = -1)            

            # taken directly from the deprecated GPR implementation
            assert mean_function is not None
            err = Y_data_1d - mean_function(X_data_1d)

            Kmm = kernel(X_data_1d)
            jitter = default_jitter() * tf.eye(Kmm.shape[0], dtype=default_float())
            Kmm += jitter
            Knn = kernel(Xnew, full_cov=full_cov)
            Kmn = kernel(X_data_1d, Xnew)
            Kmm_plus_s = add_noise_cov(Kmm, likelihood_variance)

            mean, var = base_conditional(
                Kmn, Kmm_plus_s, Knn, err, full_cov=full_cov, white=False
            ) # [N, P], [N, P] or [P, N, N]
            mean_list.append(mean), var_list.append(var)

        mean = tf.concat(mean_list, axis = -1)
        if full_cov:
            var = tf.concat(var_list, axis = 0)
        else:
            var = tf.concat(var_list, axis = 1)
        return mean, var
    
    
class MultivariateSGPRPosterior(MultivariateAbstractPosterior):
    """
        Code based on gpflow.posteriors.SGPRPosterior @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
        
        Added multivariate functionality
    """
    def __init__(
        self,
        kernel: Kernel,
        data: RegressionData,
        inducing_variable: InducingPoints,
        num_latent_gps: int,
        likelihood: Likelihood,
        mean_function: MeanFunction,
        *,
        precompute_cache: Optional[PrecomputeCacheType],
    ) -> None:
        X, Y = data
        super().__init__(kernel, X, mean_function=mean_function)
        self.Y_data = Y
        self.num_latent_gps = Y.shape[-1]
        
        self.likelihood = likelihood
        self.inducing_variable = inducing_variable
        self.num_latent_gps = num_latent_gps
        
        if precompute_cache is not None:
            self.update_cache(precompute_cache)

    def _precompute(self) -> Tuple[PrecomputedValue, ...]:
        Qinv_list, alpha_list = [], [] 
        for idx in range(self.num_latent_gps):
            likelihood = self.likelihood.likelihoods[idx]
            likelihood_variance = likelihood.variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]
            X_data_1d, Y_data_1d = self.X_data, tf.expand_dims(self.Y_data[..., idx], axis = -1)
        
            # taken directly from the deprecated SGPR implementation
            num_inducing = inducing_variable.num_inducing
            assert mean_function is not None
            err = Y_data_1d - mean_function(X_data_1d)
            kuf = Kuf(inducing_variable, kernel, X_data_1d)
            kuu = Kuu(inducing_variable, kernel, jitter=default_jitter())
            sigma = tf.sqrt(likelihood_variance)
            L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
            A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
            B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
                num_inducing, dtype=default_float()
            )  # cache qinv
            LB = tf.linalg.cholesky(B)  # cache alpha
            Aerr = tf.linalg.matmul(A, err)
            c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma  # cache alpha

            # get intermediate variables
            Linv = tf.linalg.triangular_solve(L, tf.eye(num_inducing, dtype=default_float()))
            LBinv = tf.linalg.triangular_solve(LB, tf.eye(num_inducing, dtype=default_float()))
            Binv = tf.linalg.inv(B)  # naive...can do better?
            tmp = tf.eye(num_inducing, dtype=default_float()) - Binv

            # calculate cached values
            LinvT = tf.transpose(Linv)
            alpha = LinvT @ tf.transpose(LBinv) @ c
            Qinv = LinvT @ tmp @ Linv
            Qinv_list.append(Qinv[..., None]), alpha_list.append(alpha[..., None])
            
        Qinv = tf.concat(Qinv_list, axis = -1)
        alpha = tf.concat(alpha_list, axis = -1)
        return (PrecomputedValue(alpha, (False, False, False)), PrecomputedValue(Qinv, (False, False, False)))

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
        alpha, Qinv = cache

        mean_list, var_list = [], []
        for idx in range(self.num_latent_gps):
            likelihood = self.likelihood.likelihoods[idx]
            likelihood_variance = likelihood.variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]
            X_data_1d, Y_data_1d = self.X_data, tf.expand_dims(self.Y_data[..., idx], axis = -1)
            
            Kus = Kuf(inducing_variable, kernel, Xnew)
            Knn = kernel(Xnew, full_cov=full_cov)

            Ksu = tf.transpose(Kus)
            mean = Ksu @ alpha[..., idx]

            if full_cov:
                var = Knn - Ksu @ Qinv[..., idx] @ Kus
                var = tf.tile(var[None, ...], [1, 1, 1])  # [P, N, N]
            else:
                Kfu_Qinv_Kuf = tf.reduce_sum(Kus * tf.matmul(Qinv[..., idx], Kus), axis=-2)
                var = Knn - Kfu_Qinv_Kuf
                var = tf.tile(var[:, None], [1, 1])
            mean_list.append(mean), var_list.append(var)
                                             
        # concat
        mean = tf.concat(mean_list, axis = -1)
        if full_cov:
            var = tf.concat(var_list, axis = 0)
        else:
            var = tf.concat(var_list, axis = 1)
        return mean, var

    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. Does not make use of caching
        """
        mean_list, var_list = [], []
        for idx in range(self.num_latent_gps):
            likelihood_variance = self.likelihood.likelihoods[idx].variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]
            X_data_1d, Y_data_1d = self.X_data, tf.expand_dims(self.Y_data[..., idx], axis = -1)
                                             
            # taken directly from the deprecated SGPR implementation
            num_inducing = inducing_variable.num_inducing
            assert mean_function is not None
            err = Y_data_1d - mean_function(X_data_1d)
            kuf = Kuf(inducing_variable, kernel, X_data_1d)
            kuu = Kuu(inducing_variable, kernel, jitter=default_jitter())
            Kus = Kuf(inducing_variable, kernel, Xnew)
            sigma = tf.sqrt(likelihood_variance)
            L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
            A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
            B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
                num_inducing, dtype=default_float()
            )  # cache qinv
            LB = tf.linalg.cholesky(B)  # cache alpha
            Aerr = tf.linalg.matmul(A, err)
            c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma  # cache alpha
            tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
            tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
            mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
            if full_cov:
                var = (
                    kernel(Xnew)
                    + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                    - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
                )
                var = tf.tile(var[None, ...], [1, 1, 1])  # [P, N, N]
            else:
                var = (
                    kernel(Xnew, full_cov=False)
                    + tf.reduce_sum(tf.square(tmp2), 0)
                    - tf.reduce_sum(tf.square(tmp1), 0)
                )
                var = tf.tile(var[:, None], [1, 1])
            mean_list.append(mean), var_list.append(var)

        mean = tf.concat(mean_list, axis = -1)
        if full_cov:
            var = tf.concat(var_list, axis = 0)
        else:
            var = tf.concat(var_list, axis = 1)
        return mean, var
    
class MultivariateGPRFITCPosterior(MultivariateAbstractPosterior):
    """
        Code based on gpflow.posteriors.GPRFITCPosterior @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
        
        Added multivariate functionality
    """

    def __init__(
        self,
        kernel: Kernel,
        data: RegressionData,
        inducing_variable: InducingPoints,
        num_latent_gps: int,
        likelihood: Likelihood,
        mean_function: MeanFunction,
        *,
        precompute_cache: Optional[PrecomputeCacheType],
    ) -> None:
        X, Y = data
        super().__init__(kernel, X, mean_function=mean_function)
        self.Y_data = Y
        self.num_latent_gps = Y.shape[-1]
        
        self.likelihood = likelihood
        self.inducing_variable = inducing_variable
        self.num_latent_gps = num_latent_gps
        
        if precompute_cache is not None:
            self.update_cache(precompute_cache)

    def _precompute(self) -> Tuple[PrecomputedValue, ...]:
        Qinv_list, alpha_list = [], [] 
        for idx in range(self.num_latent_gps):
            likelihood = self.likelihood.likelihoods[idx]
            likelihood_variance = likelihood.variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]
            X_data_1d, Y_data_1d = self.X_data, tf.expand_dims(self.Y_data[..., idx], axis = -1)
        
            # taken directly from the deprecated SGPR implementation
            num_inducing = inducing_variable.num_inducing
            assert mean_function is not None
            err = Y_data_1d - mean_function(X_data_1d)
            kuf = Kuf(inducing_variable, kernel, X_data_1d)
            kuu = Kuu(inducing_variable, kernel, jitter=default_jitter())
            sigma = tf.sqrt(likelihood_variance)
            L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
            A = tf.linalg.triangular_solve(L, kuf, lower=True)
            
            Kdiag = kernel(X_data_1d, full_cov=False)
            diagQff = tf.reduce_sum(tf.square(A), 0)
            nu = Kdiag - diagQff + likelihood_variance
        
            B = tf.linalg.matmul(A / nu, A, transpose_b=True) + tf.eye(
                num_inducing, dtype=default_float()
            )  # cache qinv
            LB = tf.linalg.cholesky(B)  # cache alpha
            beta = err / tf.expand_dims(nu, 1)  # size [N, R]
            Aerr = tf.linalg.matmul(A, beta)
            c = tf.linalg.triangular_solve(LB, Aerr, lower=True)

            # get intermediate variables
            Linv = tf.linalg.triangular_solve(L, tf.eye(num_inducing, dtype=default_float()))
            LBinv = tf.linalg.triangular_solve(LB, tf.eye(num_inducing, dtype=default_float()))
            Binv = tf.linalg.inv(B)  # naive...can do better?
            tmp = tf.eye(num_inducing, dtype=default_float()) - Binv

            # calculate cached values
            LinvT = tf.transpose(Linv)
            alpha = LinvT @ tf.transpose(LBinv) @ c
            Qinv = LinvT @ tmp @ Linv
            Qinv_list.append(Qinv[..., None]), alpha_list.append(alpha[..., None])
            
        Qinv = tf.concat(Qinv_list, axis = -1)
        alpha = tf.concat(alpha_list, axis = -1)
        return (PrecomputedValue(alpha, (False, False, False)), PrecomputedValue(Qinv, (False, False, False)))

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
        alpha, Qinv = cache

        mean_list, var_list = [], []
        for idx in range(self.num_latent_gps):
            likelihood = self.likelihood.likelihoods[idx]
            likelihood_variance = likelihood.variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]
            X_data_1d, Y_data_1d = self.X_data, tf.expand_dims(self.Y_data[..., idx], axis = -1)
            
            Kus = Kuf(inducing_variable, kernel, Xnew)
            Knn = kernel(Xnew, full_cov=full_cov)

            Ksu = tf.transpose(Kus)
            mean = Ksu @ alpha[..., idx]

            if full_cov:
                var = Knn - Ksu @ Qinv[..., idx] @ Kus
                var = tf.tile(var[None, ...], [1, 1, 1])  # [P, N, N]
            else:
                Kfu_Qinv_Kuf = tf.reduce_sum(Kus * tf.matmul(Qinv[..., idx], Kus), axis=-2)
                var = Knn - Kfu_Qinv_Kuf
                var = tf.tile(var[:, None], [1, 1])
            mean_list.append(mean), var_list.append(var)
                                             
        # concat
        mean = tf.concat(mean_list, axis = -1)
        if full_cov:
            var = tf.concat(var_list, axis = 0)
        else:
            var = tf.concat(var_list, axis = 1)
        return mean, var

    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. Does not make use of caching
        """
        mean_list, var_list = [], []
        for idx in range(self.num_latent_gps):
            likelihood_variance = self.likelihood.likelihoods[idx].variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]
            X_data_1d, Y_data_1d = self.X_data, tf.expand_dims(self.Y_data[..., idx], axis = -1)
                                             
            # taken directly from the deprecated SGPR implementation
            num_inducing = inducing_variable.num_inducing
            assert mean_function is not None
            err = Y_data_1d - mean_function(X_data_1d)
            kuf = Kuf(inducing_variable, kernel, X_data_1d)
            kuu = Kuu(inducing_variable, kernel, jitter=default_jitter())
            Kus = Kuf(inducing_variable, kernel, Xnew)
            sigma = tf.sqrt(likelihood_variance)
            L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
            A = tf.linalg.triangular_solve(L, kuf, lower=True)
            
            Kdiag = kernel(X_data_1d, full_cov=False)
            diagQff = tf.reduce_sum(tf.square(A), 0)
            nu = Kdiag - diagQff + likelihood_variance
        
            B = tf.linalg.matmul(A / nu, A, transpose_b=True) + tf.eye(
                num_inducing, dtype=default_float()
            )  # cache qinv
            LB = tf.linalg.cholesky(B)  # cache alpha
            beta = err / tf.expand_dims(nu, 1)  # size [N, R]
            Aerr = tf.linalg.matmul(A, beta)

            c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
            tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
            tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
            mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
            if full_cov:
                var = (
                    kernel(Xnew)
                    + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                    - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
                )
                var = tf.tile(var[None, ...], [1, 1, 1])  # [P, N, N]
            else:
                var = (
                    kernel(Xnew, full_cov=False)
                    + tf.reduce_sum(tf.square(tmp2), 0)
                    - tf.reduce_sum(tf.square(tmp1), 0)
                )
                var = tf.tile(var[:, None], [1, 1])
            mean_list.append(mean), var_list.append(var)

        mean = tf.concat(mean_list, axis = -1)
        if full_cov:
            var = tf.concat(var_list, axis = 0)
        else:
            var = tf.concat(var_list, axis = 1)
        return mean, var
    
class BasePosterior(MultivariateAbstractPosterior):
    """
        Code based on gpflow.posteriors.BasePosterior @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
    """
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable: InducingVariables,
        q_mu: tf.Tensor,
        q_sqrt: tf.Tensor,
        whiten: bool = True,
        mean_function: Optional[mean_functions.MeanFunction] = None,
        *,
        precompute_cache: Optional[PrecomputeCacheType],
    ):

        super().__init__(kernel, inducing_variable, mean_function=mean_function)
        self.whiten = whiten
        self._set_qdist(q_mu, q_sqrt)
        self.likelihood = likelihood
        
        if precompute_cache is not None:
            self.update_cache(precompute_cache)

    @property
    def q_mu(self) -> tf.Tensor:
        return self._q_dist.q_mu

    @property
    def q_sqrt(self) -> tf.Tensor:
        return self._q_dist.q_sqrt

    def _set_qdist(self, q_mu: TensorType, q_sqrt: TensorType) -> tf.Tensor:
        if q_sqrt is None:
            self._q_dist = _DeltaDist(q_mu)
        elif len(q_sqrt.shape) == 2:  # q_diag
            self._q_dist = _DiagNormal(q_mu, q_sqrt)
        else:
            self._q_dist = _MvNormal(q_mu, q_sqrt)

    def _precompute(self) -> Tuple[PrecomputedValue, ...]:
        Kuu = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [(R), M, M]
        q_mu = self._q_dist.q_mu

        if Kuu.shape.ndims == 4:
            ML = tf.reduce_prod(tf.shape(Kuu)[:2])
            Kuu = tf.reshape(Kuu, [ML, ML])
        if Kuu.shape.ndims == 3:
            q_mu = tf.linalg.adjoint(self._q_dist.q_mu)[..., None]  # [..., R, M, 1]
        L = tf.linalg.cholesky(Kuu)

        if not self.whiten:
            # alpha = Kuu⁻¹ q_mu
            alpha = tf.linalg.cholesky_solve(L, q_mu)
        else:
            # alpha = L⁻ᵀ q_mu
            alpha = tf.linalg.triangular_solve(L, q_mu, adjoint=True)
        # predictive mean = Kfu alpha
        # predictive variance = Kff - Kfu Qinv Kuf
        # S = q_sqrt q_sqrtᵀ
        I = tf.eye(tf.shape(L)[-1], dtype=L.dtype)
        if isinstance(self._q_dist, _DeltaDist):
            B = I
        else:
            if not self.whiten:
                # Qinv = Kuu⁻¹ - Kuu⁻¹ S Kuu⁻¹
                #      = Kuu⁻¹ - L⁻ᵀ L⁻¹ S L⁻ᵀ L⁻¹
                #      = L⁻ᵀ (I - L⁻¹ S L⁻ᵀ) L⁻¹
                #      = L⁻ᵀ B L⁻¹
                if isinstance(self._q_dist, _DiagNormal):
                    q_sqrt = tf.linalg.diag(tf.linalg.adjoint(self._q_dist.q_sqrt))
                elif isinstance(self._q_dist, _MvNormal):
                    q_sqrt = self._q_dist.q_sqrt
                Linv_qsqrt = tf.linalg.triangular_solve(L, q_sqrt)
                Linv_cov_u_LinvT = tf.matmul(Linv_qsqrt, Linv_qsqrt, transpose_b=True)
            else:
                if isinstance(self._q_dist, _DiagNormal):
                    Linv_cov_u_LinvT = tf.linalg.diag(tf.linalg.adjoint(self._q_dist.q_sqrt ** 2))
                elif isinstance(self._q_dist, _MvNormal):
                    q_sqrt = self._q_dist.q_sqrt
                    Linv_cov_u_LinvT = tf.matmul(q_sqrt, q_sqrt, transpose_b=True)
                # Qinv = Kuu⁻¹ - L⁻ᵀ S L⁻¹
                # Linv = (L⁻¹ I) = solve(L, I)
                # Kinv = Linvᵀ @ Linv
            B = I - Linv_cov_u_LinvT
        LinvT_B = tf.linalg.triangular_solve(L, B, adjoint=True)
        B_Linv = tf.linalg.adjoint(LinvT_B)
        Qinv = tf.linalg.triangular_solve(L, B_Linv, adjoint=True)

        M, L = tf.unstack(tf.shape(self._q_dist.q_mu), num=2)
        Qinv = tf.broadcast_to(Qinv, [L, M, M])

        tf.debugging.assert_shapes(
            [
                (Qinv, ["L", "M", "M"]),
            ]
        )

        return (PrecomputedValue(alpha, (False, False, False)), PrecomputedValue(Qinv, (False, False, False)))


class IndependentPosterior(BasePosterior):
    """
        Code based on gpflow.posteriors.IndependentPosterior @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
    """
    def _post_process_mean_and_cov(
        self, mean: TensorType, cov: TensorType, full_cov: bool, full_output_cov: bool
    ) -> MeanAndVariance:
        return mean, expand_independent_outputs(cov, full_cov, full_output_cov)

    def _get_Kff(self, Xnew: TensorType, full_cov: bool) -> tf.Tensor:

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        if isinstance(self.kernel, (kernels.SeparateIndependent, kernels.IndependentLatent)):
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] -- this is what we want
            # else: [N, P] instead of [P, N] as we get from the explicit stack below
            Kff = tf.stack([k(Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)
        elif isinstance(self.kernel, kernels.MultioutputKernel):
            # effectively, SharedIndependent path
            Kff = self.kernel.kernel(Xnew, full_cov=full_cov)
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] instead of [N, N]
            # else: [N, P] instead of [N]
        else:
            # standard ("single-output") kernels
            Kff = self.kernel(Xnew, full_cov=full_cov)  # [N, N] if full_cov else [N]

        return Kff

    def _conditional_with_precompute(
        self,
        cache: Tuple[tf.Tensor, ...],
        Xnew: TensorType,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        # Qinv: [L, M, M]
        # alpha: [M, L]
        alpha, Qinv = cache

        Kuf = covariances.Kuf(self.X_data, self.kernel, Xnew)  # [(R), M, N]
        Kff = self._get_Kff(Xnew, full_cov)

        mean = tf.matmul(Kuf, alpha, transpose_a=True)
        if Kuf.shape.ndims == 3:
            mean = tf.linalg.adjoint(tf.squeeze(mean, axis=-1))

        if full_cov:
            Kfu_Qinv_Kuf = tf.matmul(Kuf, Qinv @ Kuf, transpose_a=True)
            cov = Kff - Kfu_Qinv_Kuf
        else:
            # [Aᵀ B]_ij = Aᵀ_ik B_kj = A_ki B_kj
            # TODO check whether einsum is faster now?
            Kfu_Qinv_Kuf = tf.reduce_sum(Kuf * tf.matmul(Qinv, Kuf), axis=-2)
            cov = Kff - Kfu_Qinv_Kuf
            cov = tf.linalg.adjoint(cov)

        return self._post_process_mean_and_cov(mean, cov, full_cov, full_output_cov)


class IndependentPosteriorSingleOutput(IndependentPosterior):
    """
        Code based on gpflow.posteriors.IndependentPosteriorSingleOutput @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
    """
    # could almost be the same as IndependentPosteriorMultiOutput ...
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following
        # line:
        Knn = self.kernel(Xnew, full_cov=full_cov)

        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = covariances.Kuf(self.X_data, self.kernel, Xnew)  # [M, N]

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


class IndependentPosteriorMultiOutput(IndependentPosterior):
    """
        Code based on gpflow.posteriors.IndependentPosteriorMultiOutput @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
    """
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        if isinstance(self.X_data, SharedIndependentInducingVariables) and isinstance(
            self.kernel, kernels.SharedIndependent
        ):
            # same as IndependentPosteriorSingleOutput except for following line
            Knn = self.kernel.kernel(Xnew, full_cov=full_cov)
            # we don't call self.kernel() directly as that would do unnecessary tiling

            Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
            Kmn = covariances.Kuf(self.X_data, self.kernel, Xnew)  # [M, N]

            fmean, fvar = base_conditional(
                Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
            )  # [N, P],  [P, N, N] or [N, P]
        else:
            # this is the messy thing with tf.map_fn, cleaned up by the
            # st/clean_up_broadcasting_conditionals branch

            # Following are: [P, M, M]  -  [P, M, N]  -  [P, N](x N)
            Kmms = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [P, M, M]
            Kmns = covariances.Kuf(self.X_data, self.kernel, Xnew)  # [P, M, N]
            if isinstance(self.kernel, kernels.Combination):
                kernel_list = self.kernel.kernels
            else:
                kernel_list = [self.kernel.kernel] * len(self.X_data.inducing_variable_list)
            Knns = tf.stack(
                [k.K(Xnew) if full_cov else k.K_diag(Xnew) for k in kernel_list], axis=0
            )

            fmean, fvar = separate_independent_conditional_implementation(
                Kmns,
                Kmms,
                Knns,
                self.q_mu,
                q_sqrt=self.q_sqrt,
                full_cov=full_cov,
                white=self.whiten,
            )

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


class LinearCoregionalizationPosterior(IndependentPosteriorMultiOutput):
    """
        Code based on gpflow.posteriors.LinearCoregionalizationPosterior @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
    """
    def _post_process_mean_and_cov(
        self, mean: TensorType, cov: TensorType, full_cov: bool, full_output_cov: bool
    ) -> MeanAndVariance:
        """
        mean: [N, L]
        cov: [L, N, N] or [N, L]
        """
        cov = expand_independent_outputs(cov, full_cov, full_output_cov=False)
        mean, cov = mix_latent_gp(self.kernel.W, mean, cov, full_cov, full_output_cov)
        return mean, cov


class FullyCorrelatedPosterior(BasePosterior):
    """
        Code based on gpflow.posteriors.FullyCorrelatedPosterior @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
    """
    def _conditional_with_precompute(
        self,
        cache: Tuple[tf.Tensor, ...],
        Xnew: TensorType,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        # Qinv: [L, M, M]
        # alpha: [M, L]
        alpha, Qinv = cache

        Kuf = covariances.Kuf(self.X_data, self.kernel, Xnew)
        assert Kuf.shape.ndims == 4
        M, L, N, K = tf.unstack(tf.shape(Kuf), num=Kuf.shape.ndims, axis=0)
        Kuf = tf.reshape(Kuf, (M * L, N * K))

        kernel: kernels.MultioutputKernel = self.kernel
        Kff = kernel(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        # full_cov=True and full_output_cov=True: [N, P, N, P]
        # full_cov=True and full_output_cov=False: [P, N, N]
        # full_cov=False and full_output_cov=True: [N, P, P]
        # full_cov=False and full_output_cov=False: [N, P]
        if full_cov == full_output_cov:
            new_shape = (N * K, N * K) if full_cov else (N * K,)
            Kff = tf.reshape(Kff, new_shape)

        N = tf.shape(Xnew)[0]
        K = tf.shape(Kuf)[-1] // N

        mean = tf.matmul(Kuf, alpha, transpose_a=True)
        if Kuf.shape.ndims == 3:
            mean = tf.linalg.adjoint(tf.squeeze(mean, axis=-1))

        if not full_cov and not full_output_cov:
            # fully diagonal case in both inputs and outputs
            # [Aᵀ B]_ij = Aᵀ_ik B_kj = A_ki B_kj
            # TODO check whether einsum is faster now?
            Kfu_Qinv_Kuf = tf.reduce_sum(Kuf * tf.matmul(Qinv, Kuf), axis=-2)
        else:
            Kfu_Qinv_Kuf = tf.matmul(Kuf, Qinv @ Kuf, transpose_a=True)
            if not (full_cov and full_output_cov):
                # diagonal in either inputs or outputs
                new_shape = tf.concat([tf.shape(Kfu_Qinv_Kuf)[:-2], (N, K, N, K)], axis=0)
                Kfu_Qinv_Kuf = tf.reshape(Kfu_Qinv_Kuf, new_shape)
                if full_cov:
                    # diagonal in outputs: move outputs to end
                    tmp = tf.linalg.diag_part(tf.einsum("...ijkl->...ikjl", Kfu_Qinv_Kuf))
                elif full_output_cov:
                    # diagonal in inputs: move inputs to end
                    tmp = tf.linalg.diag_part(tf.einsum("...ijkl->...jlik", Kfu_Qinv_Kuf))
                Kfu_Qinv_Kuf = tf.einsum("...ijk->...kij", tmp)  # move diagonal dim to [-3]
        cov = Kff - Kfu_Qinv_Kuf

        if not full_cov and not full_output_cov:
            cov = tf.linalg.adjoint(cov)

        mean = tf.reshape(mean, (N, K))
        if full_cov == full_output_cov:
            cov_shape = (N, K, N, K) if full_cov else (N, K)
        else:
            cov_shape = (K, N, N) if full_cov else (N, K, K)
        cov = tf.reshape(cov, cov_shape)

        return mean, cov

    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, L, M, L]
        Kmn = covariances.Kuf(self.X_data, self.kernel, Xnew)  # [M, L, N, P]
        kernel: kernels.MultioutputKernel = self.kernel
        Knn = kernel(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )  # [N, P](x N)x P  or  [N, P](x P)

        M, L, N, K = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)
        Kmm = tf.reshape(Kmm, (M * L, M * L))

        if full_cov == full_output_cov:
            Kmn = tf.reshape(Kmn, (M * L, N * K))
            Knn = tf.reshape(Knn, (N * K, N * K)) if full_cov else tf.reshape(Knn, (N * K,))
            mean, cov = base_conditional(
                Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
            )  # [K, 1], [1, K](x NK)
            mean = tf.reshape(mean, (N, K))
            cov = tf.reshape(cov, (N, K, N, K) if full_cov else (N, K))
        else:
            Kmn = tf.reshape(Kmn, (M * L, N, K))
            mean, cov = fully_correlated_conditional(
                Kmn,
                Kmm,
                Knn,
                self.q_mu,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                q_sqrt=self.q_sqrt,
                white=self.whiten,
            )
        return mean, cov


class FallbackIndependentLatentPosterior(FullyCorrelatedPosterior):  # XXX
    """
        Code based on gpflow.posteriors.FallbackIndependentLatentPosterior @
        @ARTICLE{GPflow2017,
          author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
            Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
            Ghahramani, Zoubin and Hensman, James},
          title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
          journal = {Journal of Machine Learning Research},
          year    = {2017},
          month = {apr},
          volume  = {18},
          number  = {40},
          pages   = {1-6},
          url     = {http://jmlr.org/papers/v18/16-537.html}
        }
    """
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [L, M, M]
        Kmn = covariances.Kuf(self.X_data, self.kernel, Xnew)  # [M, L, N, P]
        kernel: kernels.IndependentLatent = self.kernel
        Knn = kernel(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )  # [N, P](x N)x P  or  [N, P](x P)

        return independent_interdomain_conditional(
            Kmn,
            Kmm,
            Knn,
            self.q_mu,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            q_sqrt=self.q_sqrt,
            white=self.whiten,
        )


get_posterior_class = Dispatcher("get_posterior_class")


@get_posterior_class.register(kernels.Kernel, InducingVariables)
def _get_posterior_base_case(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    # independent single output
    return IndependentPosteriorSingleOutput


@get_posterior_class.register(kernels.MultioutputKernel, InducingPoints)
def _get_posterior_fully_correlated_mo(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    return FullyCorrelatedPosterior


@get_posterior_class.register(
    (kernels.SharedIndependent, kernels.SeparateIndependent),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_independent_mo(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    # independent multi-output
    return IndependentPosteriorMultiOutput


@get_posterior_class.register(
    kernels.IndependentLatent,
    (FallbackSeparateIndependentInducingVariables, FallbackSharedIndependentInducingVariables),
)
def _get_posterior_independentlatent_mo_fallback(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    return FallbackIndependentLatentPosterior


@get_posterior_class.register(
    kernels.LinearCoregionalization,
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_linearcoregionalization_mo_efficient(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    # Linear mixing---efficient multi-output
    return LinearCoregionalizationPosterior


def create_posterior(
    kernel: Kernel,
    likelihood: Likelihood,
    inducing_variable: InducingVariables,
    q_mu: TensorType,
    q_sqrt: TensorType,
    whiten: bool,
    mean_function: Optional[MeanFunction] = None,
    precompute_cache: Union[PrecomputeCacheType, str, None] = PrecomputeCacheType.TENSOR,
) -> BasePosterior:
    posterior_class = get_posterior_class(kernel, inducing_variable)
    precompute_cache = _validate_precompute_cache_type(precompute_cache)
    return posterior_class(  # type: ignore
        kernel,
        likelihood,
        inducing_variable,
        q_mu,
        q_sqrt,
        whiten,
        mean_function,
        precompute_cache=precompute_cache,
    )