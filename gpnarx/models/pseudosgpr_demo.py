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
from .sgpr import MultivariateSGPRBase_deprecated
from .narx import NARX, GP_NARX

class MultivariatePSEUDOSGPR_DEMO(MultivariateSGPRBase_deprecated):
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

    def predict_f(
        self, 
        Xnew: InputData, 
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        full_cov: bool = True,
        full_output_cov: bool = False,    
        naive = False
    )  -> MeanAndVariance:
        mean_list, var_list = [], []
        for idx in range(Xnew.shape[0]):
            Xiter = tf.expand_dims(Xnew[idx, ...], axis = 0)
            mean, var = self.predict_f_1d(
                Xnew = Xiter,
                observed_data = observed_data,
                observed_latent = observed_latent,    
                naive = naive
            )
            mean_list.append(mean), var_list.append(var)   
        mean = tf.concat(mean_list, axis = 0)
        var = tf.concat(var_list, axis = 0)
        return mean, var       
    
    def predict_f_1d(
        self, 
        Xnew: InputData, 
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        naive = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        assert Xnew.shape[0] == 1
        full_cov = False
        
        X_data, Y_data = self.data

        if observed_data is not None:
            X_data_obs, Y_data_obs = observed_data
            X_all = tf.concat([X_data, X_data_obs], axis = 0)
            Y_all = tf.concat([Y_data, Y_data_obs], axis = 0)
        else:
            X_all, Y_all = X_data, Y_data
            
        mean_list, var_list = [], []
        for idx in range(self.num_latent_gps):
            likelihood_variance = self.likelihood.likelihoods[idx].variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            inducing_variable = self.inducing_variable.inducing_variable_list[idx]
            X_data_1d, Y_data_1d = X_all, tf.expand_dims(Y_all[..., idx], axis = -1)

            if observed_data is not None:
                if observed_latent:
                    fill_value = tf.constant(default_jitter(), dtype = default_float())
                else:
                    fill_value = likelihood_variance
                nu = tf.concat([
                    tf.fill(dims = [X_data.shape[0],], value = likelihood_variance), 
                    tf.fill(dims = [X_data_obs.shape[0],], value = fill_value)
                ], axis = 0)
                if not naive:
                    inducing_variable_Z = tf.concat([
                        inducing_variable.Z,
                        Xnew
                    ], axis = 0)
                    num_inducing = inducing_variable.num_inducing + 1
                else:
                    inducing_variable_Z = inducing_variable.Z
                    num_inducing = inducing_variable.num_inducing
            else:
                nu = tf.fill(dims = [X_data.shape[0],], value = likelihood_variance)
                inducing_variable_Z = inducing_variable.Z
                num_inducing = inducing_variable.num_inducing
            
            assert mean_function is not None
            err = Y_data_1d - mean_function(X_data_1d)
            kuf = kernel(inducing_variable_Z, X_data_1d)
            kuu = kernel(inducing_variable_Z, full_cov = True)
            Kus = kernel(inducing_variable_Z, Xnew)
            
            jitter = default_jitter() * tf.eye(kuu.shape[0], dtype=default_float())
            L = tf.linalg.cholesky(kuu + jitter)  # cache alpha, qinv
            A = tf.linalg.triangular_solve(L, kuf, lower=True)
            
            B = tf.linalg.matmul(A / tf.sqrt(nu), A / tf.sqrt(nu), transpose_b=True) + tf.eye(
                num_inducing, dtype=default_float()
            )  # cache qinv
            jitter = default_jitter() * tf.eye(B.shape[0], dtype=default_float())
            LB = tf.linalg.cholesky(B + jitter)  # cache alpha
            
            beta = err / tf.expand_dims(nu, 1)  # size [N, R]
            Aerr = tf.linalg.matmul(A, beta)

            c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
            tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
            tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
            mean = tf.linalg.matmul(tmp2, c, transpose_a=True)

            var = (
                kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, 1])
            mean_list.append(mean), var_list.append(var)

        mean = tf.concat(mean_list, axis = -1)
        var = tf.concat(var_list, axis = -1)
        return mean, var

    def predict_f_samples(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = None,
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        full_cov: bool = True,
        full_output_cov: bool = False,
        naive = False
    ) -> tf.Tensor:

        assert full_cov == False
        assert full_output_cov == False

        # check below for shape info
        mean, cov = self.predict_f(
            Xnew, 
            observed_data=observed_data,
            observed_latent=observed_latent,
            full_cov=full_cov, 
            full_output_cov=full_output_cov,
            naive = naive
        )
           
        # mean: [..., N, P]
        # cov: [..., N, P] or [..., N, P, P]
        samples = sample_mvn(
            mean, cov, full_output_cov, num_samples=num_samples
        )  # [..., (S), N, P]
        return samples  # [..., (S), N, P]