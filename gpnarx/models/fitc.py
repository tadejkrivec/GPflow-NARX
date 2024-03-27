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

from gpflow import likelihoods
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

from .. import posteriors_multivariate, posteriors
from gpflow.models.sgpr import SGPRBase_deprecated
from gpnarx.models.sgpr import MultivariateSGPRBase_deprecated
from .narx import NARX, GP_NARX


class GPRFITC_deprecated(SGPRBase_deprecated):
    """
    This implements GP regression with the FITC approximation.
    The key reference is
    ::
      @inproceedings{Snelson06sparsegaussian,
        author = {Edward Snelson and Zoubin Ghahramani},
        title = {Sparse Gaussian Processes using Pseudo-inputs},
        booktitle = {Advances In Neural Information Processing Systems},
        year = {2006},
        pages = {1257--1264},
        publisher = {MIT press}
      }
    Implementation loosely based on code from GPML matlab library although
    obviously gradients are automatic in GPflow.
    
    Code based on gpflow.models.sgpr.GPRFITC @
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

    def common_terms(self):
        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)  # size [N, R]
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
        V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf

        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + self.likelihood.variance

        B = tf.eye(num_inducing, dtype=default_float()) + tf.linalg.matmul(
            V / nu, V, transpose_b=True
        )
        L = tf.linalg.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size [N, R]
        alpha = tf.linalg.matmul(V, beta)  # size [N, R]

        gamma = tf.linalg.triangular_solve(L, alpha, lower=True)  # size [N, R]

        return err, nu, Luu, L, alpha, beta, gamma

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.fitc_log_marginal_likelihood()

    def fitc_log_marginal_likelihood(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        # FITC approximation to the log marginal likelihood is
        # log ( normal( y | mean, K_fitc ) )
        # where K_fitc = Qff + diag( \nu )
        # where Qff = Kfu kuu^{-1} kuf
        # with \nu_i = Kff_{i,i} - Qff_{i,i} + \sigma^2

        # We need to compute the Mahalanobis term -0.5* err^T K_fitc^{-1} err
        # (summed over functions).

        # We need to deal with the matrix inverse term.
        # K_fitc^{-1} = ( Qff + \diag( \nu ) )^{-1}
        #            = ( V^T V + \diag( \nu ) )^{-1}
        # Applying the Woodbury identity we obtain
        #            = \diag( \nu^{-1} ) - \diag( \nu^{-1} ) V^T ( I + V \diag( \nu^{-1} ) V^T )^{-1) V \diag(\nu^{-1} )
        # Let \beta =  \diag( \nu^{-1} ) err
        # and let \alpha = V \beta
        # then Mahalanobis term = -0.5* ( \beta^T err - \alpha^T Solve( I + V \diag( \nu^{-1} ) V^T, alpha ) )

        err, nu, Luu, L, alpha, beta, gamma = self.common_terms()

        mahalanobisTerm = -0.5 * tf.reduce_sum(
            tf.square(err) / tf.expand_dims(nu, 1)
        ) + 0.5 * tf.reduce_sum(tf.square(gamma))

        # We need to compute the log normalizing term -N/2 \log 2 pi - 0.5 \log \det( K_fitc )

        # We need to deal with the log determinant term.
        # \log \det( K_fitc ) = \log \det( Qff + \diag( \nu ) )
        #                    = \log \det( V^T V + \diag( \nu ) )
        # Applying the determinant lemma we obtain
        #                    = \log [ \det \diag( \nu ) \det( I + V \diag( \nu^{-1} ) V^T ) ]
        #                    = \log [ \det \diag( \nu ) ] + \log [ \det( I + V \diag( \nu^{-1} ) V^T ) ]

        constantTerm = -0.5 * self.num_data * tf.math.log(tf.constant(2.0 * np.pi, default_float()))
        logDeterminantTerm = -0.5 * tf.reduce_sum(tf.math.log(nu)) - tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(L))
        )
        logNormalizingTerm = constantTerm + logDeterminantTerm

        return mahalanobisTerm + logNormalizingTerm * self.num_latent_gps

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        _, _, Luu, L, _, _, gamma = self.common_terms()
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, N]

        w = tf.linalg.triangular_solve(Luu, Kus, lower=True)  # [M, N]

        tmp = tf.linalg.triangular_solve(tf.transpose(L), gamma, lower=False)
        mean = tf.linalg.matmul(w, tmp, transpose_a=True) + self.mean_function(Xnew)
        intermediateA = tf.linalg.triangular_solve(L, w, lower=True)

        if full_cov:
            var = (
                self.kernel(Xnew)
                - tf.linalg.matmul(w, w, transpose_a=True)
                + tf.linalg.matmul(intermediateA, intermediateA, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                - tf.reduce_sum(tf.square(w), 0)
                + tf.reduce_sum(tf.square(intermediateA), 0)
            )  # [N, P]
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean, var
    
class GPRFITC_with_posterior(GPRFITC_deprecated):
    """
    This is an implementation of GPRFITC that provides a posterior() method that
    enables caching for faster subsequent predictions.
    
    Code based on gpflow.models.sgpr.SGPR_with_posterior @
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

    def posterior(self, precompute_cache=posteriors.PrecomputeCacheType.TENSOR):
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

        return posteriors.GPRFITCPosterior(
            kernel=self.kernel,
            data=self.data,
            inducing_variable=self.inducing_variable,
            likelihood_variance=self.likelihood.variance,
            num_latent_gps=self.num_latent_gps,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        For backwards compatibility, GPR's predict_f uses the fused (no-cache)
        computation, which is more efficient during training.
        For faster (cached) prediction, predict directly from the posterior object, i.e.,:
            model.posterior().predict_f(Xnew, ...)
        """
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
    
class MultivariateGPRFITC_deprecated(MultivariateSGPRBase_deprecated):
    """
    Code based on gpflow.models.sgpr.SGPR_with_posterior @
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
    def common_terms_1d(self, idx: int):
        X_data, Y_data = self.data
        likelihood = self.likelihood.likelihoods[idx]
        kernel = self.kernel.kernels[idx]
        mean_function = self.mean_function.mean_functions[idx]
        X_data_1d, Y_data_1d = X_data, tf.expand_dims(Y_data[..., idx], axis = -1)
        inducing_variable = self.inducing_variable.inducing_variable_list[idx]
        
        num_inducing = inducing_variable.num_inducing
        err = Y_data_1d - mean_function(X_data_1d)  # size [N, R]
        Kdiag = kernel(X_data_1d, full_cov=False)
        kuf = Kuf(inducing_variable, kernel, X_data_1d)
        kuu = Kuu(inducing_variable, kernel, jitter=default_jitter())

        Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
        V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf

        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + likelihood.variance

        B = tf.eye(num_inducing, dtype=default_float()) + tf.linalg.matmul(
            V / nu, V, transpose_b=True
        )
        L = tf.linalg.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size [N, R]
        alpha = tf.linalg.matmul(V, beta)  # size [N, R]

        gamma = tf.linalg.triangular_solve(L, alpha, lower=True)  # size [N, R]

        return err, nu, Luu, L, alpha, beta, gamma

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.fitc_log_marginal_likelihood()

    def fitc_log_marginal_likelihood(self) -> tf.Tensor:
        mll = tf.constant(0., dtype = default_float())
        for idx in range(self.num_latent_gps):
            err, nu, Luu, L, alpha, beta, gamma = self.common_terms_1d(idx = idx)

            mahalanobisTerm = -0.5 * tf.reduce_sum(
                tf.square(err) / tf.expand_dims(nu, 1)
            ) + 0.5 * tf.reduce_sum(tf.square(gamma))

            constantTerm = -0.5 * self.num_data * tf.math.log(tf.constant(2.0 * np.pi, default_float()))
            logDeterminantTerm = -0.5 * tf.reduce_sum(tf.math.log(nu)) - tf.reduce_sum(
                tf.math.log(tf.linalg.diag_part(L))
            )
            logNormalizingTerm = constantTerm + logDeterminantTerm
            mll += mahalanobisTerm 
            mll += logNormalizingTerm
        return mll
    
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
            
            kuf = Kuf(inducing_variable, kernel, X_data_1d)
            kuu = Kuu(inducing_variable, kernel, jitter=default_jitter())
            
            Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
            V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf
            
            Kdiag = kernel(X_data_1d, full_cov=False)
            diagQff = tf.reduce_sum(tf.square(V), 0)
            nu = Kdiag - diagQff + likelihood_variance
        
            sig = kuu + tf.matmul(kuf / nu, kuf, transpose_b=True)
            sig_sqrt = tf.linalg.cholesky(sig)

            sig_sqrt_kuu = tf.linalg.triangular_solve(sig_sqrt, kuu)

            cov = tf.linalg.matmul(sig_sqrt_kuu, sig_sqrt_kuu, transpose_a=True)
            err = Y_data_1d - mean_function(X_data_1d)
            beta = err / tf.expand_dims(nu, 1)  # size [N, R]
            mu = (
                tf.linalg.matmul(
                    sig_sqrt_kuu,
                    tf.linalg.triangular_solve(sig_sqrt, tf.linalg.matmul(kuf, beta)),
                    transpose_a=True,
                )
            ) + mean_function(inducing_variable.Z)
            mean_list.append(mu), var_list.append(cov[None, ...])                
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
    
    
class MultivariateGPRFITC_with_posterior(MultivariateGPRFITC_deprecated):
    """
    Code based on gpflow.models.sgpr.SGPR_with_posterior @
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

        return posteriors_multivariate.MultivariateGPRFITCPosterior(
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

class MultivariateGPRFITC(MultivariateGPRFITC_with_posterior):
    # subclassed to ensure __class__ == "SGPR"
    pass

class MultivariateGPRFITC_NARX(NARX, MultivariateGPRFITC, GP_NARX):
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
        
        # initialize MultivariateGPRFITC model
        MultivariateGPRFITC.__init__(
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