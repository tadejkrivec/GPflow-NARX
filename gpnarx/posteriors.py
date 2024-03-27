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

class GPRFITCPosterior(AbstractPosterior):
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
        likelihood_variance: Parameter,
        num_latent_gps: int,
        mean_function: MeanFunction,
        *,
        precompute_cache: Optional[PrecomputeCacheType],
    ) -> None:
        X, Y = data
        super().__init__(kernel, X, mean_function=mean_function)
        self.Y_data = Y
        self.likelihood_variance = likelihood_variance
        self.inducing_variable = inducing_variable
        self.num_latent_gps = num_latent_gps

        if precompute_cache is not None:
            self.update_cache(precompute_cache)

    def _precompute(self) -> Tuple[PrecomputedValue, ...]:
        # taken directly from the deprecated SGPR implementation
        num_inducing = self.inducing_variable.num_inducing
        assert self.mean_function is not None
        err = self.Y_data - self.mean_function(self.X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, self.X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        sigma = tf.sqrt(self.likelihood_variance)
        L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
        A = tf.linalg.triangular_solve(L, kuf, lower=True)
        Kdiag = kernel(self.X_data, full_cov=False)
        diagQff = tf.reduce_sum(tf.square(A), 0)
        nu = Kdiag - diagQff + self.likelihood_variance
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

        return (PrecomputedValue(alpha, (False, False)), PrecomputedValue(Qinv, (False, False)))

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

        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        Knn = self.kernel(Xnew, full_cov=full_cov)

        Ksu = tf.transpose(Kus)
        mean = Ksu @ alpha

        if full_cov:
            var = Knn - Ksu @ Qinv @ Kus
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            Kfu_Qinv_Kuf = tf.reduce_sum(Kus * tf.matmul(Qinv, Kus), axis=-2)
            var = Knn - Kfu_Qinv_Kuf
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean, var

    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. Does not make use of caching
        """

        # taken directly from the deprecated SGPR implementation
        num_inducing = self.inducing_variable.num_inducing
        assert self.mean_function is not None
        err = self.Y_data - self.mean_function(self.X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, self.X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma = tf.sqrt(self.likelihood_variance)
        L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
        A = tf.linalg.triangular_solve(L, kuf, lower=True)
        Kdiag = self.kernel(self.X_data, full_cov=False)
        diagQff = tf.reduce_sum(tf.square(A), 0)
        nu = Kdiag - diagQff + self.likelihood_variance
        B = tf.linalg.matmul(A / nu, A, transpose_b=True) + tf.eye(
            num_inducing, dtype=default_float()
        )
        LB = tf.linalg.cholesky(B)  # cache alpha
        beta = err / tf.expand_dims(nu, 1)  # size [N, R]
        Aerr = tf.linalg.matmul(A, beta)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean, var