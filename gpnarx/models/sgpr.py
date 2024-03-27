# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from typing import NamedTuple, Optional, Tuple, Callable

import numpy as np
import tensorflow as tf
import gpnarx
import gpflow 

from gpflow.kernels import Kernel

from gpflow import likelihoods, posteriors
from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.config import default_float, default_jitter
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.inducing_variables import InducingPoints
from gpflow.mean_functions import MeanFunction
from gpflow.likelihoods import Likelihood
from gpflow.utilities import add_noise_cov, to_default_float
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from gpflow.conditionals.util import sample_mvn

from .. import posteriors_multivariate
from .narx import NARX, GP_NARX

class MultivariateSGPRBase_deprecated(GPModel, InternalDataTrainingLossMixin):
    """
    Common base class for SGPR and GPRFITC that provides the common __init__
    and upper_bound() methods.
    
    Code based on gpflow.models.sgpr.SGPRBase_deprecated @
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
        data: RegressionData,
        kernel: Kernel,
        inducing_variable: InducingPoints,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
        noise_variance: float = 1.0,
        likelihood: Optional[Likelihood] = None
    ):
        """
        This method only works with a Gaussian likelihood, its variance is
        initialized to `noise_variance`.

        :param data: a tuple of (X, Y), where the inputs X has shape [N, D]
            and the outputs Y has shape [N, R].
        :param inducing_variable:  an InducingPoints instance or a matrix of
            the pseudo inputs Z, of shape [M, D].
        :param kernel: An appropriate GPflow kernel object.
        :param mean_function: An appropriate GPflow mean function object.
        """
        X_data, Y_data = data_input_to_tensor(data)
        
        # handle likelihood initialization
        if likelihood is None:
            likelihood = gpnarx.likelihoods.MOLikelihood([gpnarx.likelihoods.Gaussian_with_sampler(noise_variance) for _ in range(Y_data.shape[-1])])
        
        # handle mean function initialization
        if mean_function is None:
            mean_function = gpnarx.mean_functions.MOMeanfunction([gpflow.mean_functions.Zero() for _ in range(Y_data.shape[-1])])
            
        num_latent_gps = Y_data.shape[-1] if num_latent_gps is None else num_latent_gps
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = X_data, Y_data
        self.num_data = X_data.shape[0]

        self.inducing_variable: InducingPoints = inducingpoint_wrapper(inducing_variable)


class MultivariateSGPR_deprecated(MultivariateSGPRBase_deprecated):
    """
    Code based on gpflow.models.sgpr.SGPR_deprecated @
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

    CommonTensors = namedtuple("CommonTensors", ["A", "B", "LB", "AAT", "L"])

    # type-ignore is because of changed method signature:
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore
        return self.elbo()

    def _common_calculation_1d(self, idx: int) -> "SGPR.CommonTensors":
        """
        Matrices used in log-det calculation

        :return: A , B, LB, AAT with :math:`LLᵀ = Kᵤᵤ , A = L⁻¹K_{uf}/σ, AAT = AAᵀ,
            B = AAT+I, LBLBᵀ = B`
            A is M x N, B is M x M, LB is M x M, AAT is M x M
        """
        X_data_1d, _ = self.data
        sigma_sq = self.likelihood.likelihoods[idx].variance
        kernel = self.kernel.kernels[idx]
        iv = self.inducing_variable.inducing_variable_list[idx]

        kuf = Kuf(iv, kernel, X_data_1d)
        kuu = Kuu(iv, kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu)
        sigma = tf.sqrt(sigma_sq)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = add_noise_cov(AAT, tf.cast(1.0, AAT.dtype))
        LB = tf.linalg.cholesky(B)

        return self.CommonTensors(A, B, LB, AAT, L)

    def logdet_term_1d(self, common: "SGPR.CommonTensors", idx: int) -> tf.Tensor:
        r"""
        Bound from Jensen's Inequality:

        .. math::
            \log |K + σ²I| <= \log |Q + σ²I| + N * \log (1 + \textrm{tr}(K - Q)/(σ²N))

        :param common: A named tuple containing matrices that will be used
        :return: log_det, lower bound on :math:`-.5 * \textrm{output_dim} * \log |K + σ²I|`
        """
        X, Y = self.data
        sigma_sq = self.likelihood.likelihoods[idx].variance
        kernel = self.kernel.kernels[idx]
        X_data_1d, Y_data_1d = X, tf.expand_dims(Y[..., idx], axis = -1)
        
        LB = common.LB
        AAT = common.AAT

        num_data = to_default_float(tf.shape(X_data_1d)[0])
        outdim = to_default_float(tf.shape(Y_data_1d)[1])
        kdiag = kernel(X_data_1d, full_cov=False)

        # tr(K) / σ²
        trace_k = tf.reduce_sum(kdiag) / sigma_sq
        # tr(Q) / σ²
        trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))
        # tr(K - Q) / σ²
        trace = trace_k - trace_q

        # 0.5 * log(det(B))
        half_logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        # N * log(σ²)
        log_sigma_sq = num_data * tf.math.log(sigma_sq)

        logdet_k = -outdim * (half_logdet_b + 0.5 * log_sigma_sq + 0.5 * trace)
        return logdet_k

    def quad_term_1d(self, common: "SGPR.CommonTensors", idx: int) -> tf.Tensor:
        """
        :param common: A named tuple containing matrices that will be used
        :return: Lower bound on -.5 yᵀ(K + σ²I)⁻¹y
        """
        X, Y = self.data
        sigma_sq = self.likelihood.likelihoods[idx].variance
        kernel = self.kernel.kernels[idx]
        mean_function = self.mean_function.mean_functions[idx]
        X_data_1d, Y_data_1d = X, tf.expand_dims(Y[..., idx], axis = -1)
        
        A = common.A
        LB = common.LB

        err = Y_data_1d - mean_function(X_data_1d)
        sigma = tf.sqrt(sigma_sq)

        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma

        # σ⁻² yᵀy
        err_inner_prod = tf.reduce_sum(tf.square(err)) / sigma_sq
        c_inner_prod = tf.reduce_sum(tf.square(c))

        quad = -0.5 * (err_inner_prod - c_inner_prod)
        return quad

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        _, Y_data = self.data
        
        elbo = tf.constant(0., dtype = default_float())
        for idx in range(self.num_latent_gps):
            Y_data_1d = tf.expand_dims(Y_data[..., idx], axis = -1)
            common = self._common_calculation_1d(idx = idx)
            output_shape = tf.shape(Y_data_1d)
            num_data = to_default_float(output_shape[0])
            output_dim = to_default_float(output_shape[1])
            const = -0.5 * num_data * output_dim * np.log(2 * np.pi)
            logdet = self.logdet_term_1d(common, idx = idx)
            quad = self.quad_term_1d(common, idx = idx)
            elbo += const
            elbo += logdet
            elbo += quad
        return elbo
    
    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the mean and variance of q(u) = N(mu, cov), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, cov
        """
        
        X_data, Y_data = self.data
        mean_list, var_list = [], []
        
        for idx in range(self.num_latent_gps):
            likelihood_variance = self.likelihood.likelihoods[idx].variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]
            X_data_1d, Y_data_1d = X_data, tf.expand_dims(Y_data[..., idx], axis = -1)
                                             
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
            jitter = default_jitter() * tf.eye(B.shape[0], dtype=default_float())
            LB = tf.linalg.cholesky(B + jitter)  # cache alpha
            Aerr = tf.linalg.matmul(A, err)
            c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma  # cache alpha
            gamma_T = tf.linalg.triangular_solve(LB, tf.transpose(L), lower=True)
            mean = tf.linalg.matmul(gamma_T, c, transpose_a=True) + mean_function(inducing_variable.Z)
            
            var = tf.linalg.matmul(gamma_T, gamma_T, transpose_a=True)
            var = tf.tile(var[None, ...], [1, 1, 1])  # [P, N, N]

            mean_list.append(mean), var_list.append(var)

        mean = tf.concat(mean_list, axis = -1)
        var = tf.concat(var_list, axis = 0)
        return mean, var
    
    def predict_qu_samples(self, num_samples) -> tf.Tensor:
        u_mean, u_var = self.compute_qu()
        mean_for_sample = tf.linalg.adjoint(u_mean)  # [..., P, N]
        samples = sample_mvn(
            mean_for_sample, u_var, full_cov=True, num_samples=num_samples
        )  # [..., (S), P, N]
        u_samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
            
        L_list, X_data_list = [], []
        for idx in range(self.num_latent_gps):
            kernel = self.kernel.kernels[idx]
            inducing_loc = self.inducing_variable.inducing_variable_list[idx].Z
            Kmm = kernel(inducing_loc)
            jitter = default_jitter() * tf.eye(Kmm.shape[0], dtype=default_float())
            Kmm += jitter
            L = tf.linalg.cholesky(Kmm)
            L_list.append(L[..., None])
            X_data_list.append(inducing_loc[..., None])
        L = tf.concat(L_list, axis = -1)
        X_data = tf.concat(X_data_list, axis = -1)
        return X_data, u_samples, L
    
class MultivariateSGPR_with_posterior(MultivariateSGPR_deprecated):
    """
    Code based on gpflow.models.sgpr.SGPR_deprecated @
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

    def posterior(
        self,
        precompute_cache: posteriors_multivariate.PrecomputeCacheType = posteriors_multivariate.PrecomputeCacheType.TENSOR,
    ) -> posteriors_multivariate.MultivariateSGPRPosterior:
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """

        return posteriors_multivariate.MultivariateSGPRPosterior(
            kernel=self.kernel,
            data=self.data,
            inducing_variable=self.inducing_variable,
            num_latent_gps=self.num_latent_gps,
            likelihood=self.likelihood,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    def predict_f(
        self, 
        Xnew: InputData, 
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        full_cov: bool = False, 
        full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        For backwards compatibility, GPR's predict_f uses the fused (no-cache)
        computation, which is more efficient during training.
        For faster (cached) prediction, predict directly from the posterior object, i.e.,:
            model.posterior().predict_f(Xnew, ...)
        """
            
        mean, var = self.posterior(posteriors_multivariate.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, 
            observed_data = observed_data, 
            observed_latent = observed_latent,
            full_cov=full_cov, 
            full_output_cov=full_output_cov
        )
            
        # more naturaly order the shapes (output dimensions at the back)
        if full_cov:
            var = tf.transpose(var, perm = [1, 2, 0])
        else:
            var = tf.transpose(var, perm = [1, 0])
            
        return mean, var
    
    def predict_f_samples(
        self,
        Xnew: InputData,
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        num_samples: Optional[int] = None,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:

        samples = self.posterior(posteriors_multivariate.PrecomputeCacheType.NOCACHE).fused_predict_f_samples(
            Xnew = Xnew,
            num_samples = num_samples,
            observed_data = observed_data,
            observed_latent = observed_latent,
            full_cov = full_cov,
            full_output_cov = full_output_cov
        ) 
        return samples

class MultivariateSGPR(MultivariateSGPR_with_posterior):
    # subclassed to ensure __class__ == "SGPR"
    pass

class MultivariateSGPR_NARX(NARX, MultivariateSGPR, GP_NARX):
    """
        Provides a NARX extension for the static GP model
    """
    def __init__(
        self,
        data: RegressionData,
        narx_params: list, 
        inducing_variable_wrapper: Callable,
        *args, 
        **kwargs
    ):
        # convert to tensorflow array
        data = data_input_to_tensor(data)
        
        # initialize NARX model
        U, Y = data
        NARX.__init__(
            self, 
            U = U, 
            Y = Y,
            narx_params = narx_params
        )
        
        # initialize MultivariateSGPR model
        MultivariateSGPR.__init__(
            self, 
            data = (self.Z, self.Y), 
            inducing_variable = inducing_variable_wrapper(self.Z),
            *args, 
            **kwargs
        )
        
        # initialize GP-NARX
        GP_NARX.__init__(
            self
        )