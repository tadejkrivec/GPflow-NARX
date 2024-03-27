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

from typing import Optional, Tuple, Callable

import numpy as np
import tensorflow as tf
import gpflow
import gpnarx

from gpflow import default_jitter, default_float
from gpflow import kullback_leiblers, posteriors
from gpflow.base import InputData, MeanAndVariance, Parameter, RegressionData
from gpflow.conditionals import conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float
from gpflow.utilities import positive, triangular
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

from .. import posteriors_multivariate
from .narx import NARX, GP_NARX

class MultivariateSVGP_deprecated(GPModel, ExternalDataTrainingLossMixin):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    Code based on gpflow.models.svgp.SVGP @
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
        kernel,
        inducing_variable,
        *,
        mean_function=None,
        likelihood = None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        num_data=None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # handle likelihood initialization
        if likelihood is None:
            likelihood = gpnarx.likelihoods.MOLikelihood([gpnarx.likelihoods.Gaussian_with_sampler(1.0) for _ in range(num_latent_gps)])
        
        # handle mean function initialization
        if mean_function is None:
            mean_function = gpnarx.mean_functions.MOMeanfunction([gpflow.mean_functions.Zero() for _ in range(num_latent_gps)])
            
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent_gps), dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [
                    np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
                ]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent_gps = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent_gps = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def prior_kl(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl
    
    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the mean and variance of q(u) = N(mu, cov), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, cov
        """        
        mean_list, var_list = [], []        
        for idx in range(self.num_latent_gps):
            mean_function = self.mean_function.mean_functions[idx]
            Z_loc = self.inducing_variable.inducing_variable_list[idx].Z
            Kuu = self.kernel.kernels[idx](Z_loc)
            Luu = tf.linalg.cholesky(Kuu)
            mu = Luu @ self.q_mu[:, idx][:, None]  + mean_function(Z_loc)
            K = Luu @ self.q_sqrt[idx, ...]
            cov = K @ tf.transpose(K)
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
    
class MultivariateSVGP_with_posterior(MultivariateSVGP_deprecated):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    This class provides a posterior() method that enables caching for faster subsequent predictions.
    
    Code based on gpflow.models.svgp.SVGP @
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

    def posterior(self, precompute_cache=posteriors_multivariate.PrecomputeCacheType.TENSOR):
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
        return posteriors_multivariate.create_posterior(
            self.kernel,
            self.likelihood,
            self.inducing_variable,
            self.q_mu,
            self.q_sqrt,
            whiten=self.whiten,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    def predict_f(
        self, 
        Xnew: InputData, 
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        full_cov=False, 
        full_output_cov=False
    ) -> MeanAndVariance:
        """
        For backwards compatibility, SVGP's predict_f uses the fused (no-cache)
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

class MultivariateSVGP(MultivariateSVGP_with_posterior):
    # subclassed to ensure __class__ == "SVGP"
    pass

class MultivariateSVGP_NARX(NARX, MultivariateSVGP, GP_NARX):
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
        
        # initialize MultivariateSVGP model
        MultivariateSVGP.__init__(
            self, 
            inducing_variable = inducing_variable_wrapper(self.Z),
            *args, 
            **kwargs
        )
        
        # initialize GP-NARX
        GP_NARX.__init__(
            self
        )