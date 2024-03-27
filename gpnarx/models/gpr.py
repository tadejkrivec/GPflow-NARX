from typing import Optional, Tuple, Callable

import tensorflow as tf

import gpflow

import gpnarx

from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.kernels import Kernel
from gpflow.logdensities import multivariate_normal
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.utilities.model_utils import add_noise_cov
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_jitter, default_float

from .. import posteriors_multivariate
from .narx import NARX, GP_NARX

class MultivariateGPR_deprecated(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.
    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.
    The log likelihood of this model is given by
    .. math::
       \log p(Y \,|\, \mathbf f) =
            \mathcal N(Y \,|\, 0, \sigma_n^2 \mathbf{I})
    To train the model, we maximise the log _marginal_ likelihood
    w.r.t. the likelihood variance and kernel hyperparameters theta.
    The marginal likelihood is found by integrating the likelihood
    over the prior, and has the form
    .. math::
       \log p(Y \,|\, \sigma_n, \theta) =
            \mathcal N(Y \,|\, 0, \mathbf{K} + \sigma_n^2 \mathbf{I})
            
    Code based on gpflow.models.gpr.GPR @
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
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
        likelihood: Optional[Likelihood] = None
    ):
        _, Y_data = data
        
        # handle likelihood initialization
        if likelihood is None:
            likelihood = gpnarx.likelihoods.MOLikelihood([gpnarx.likelihoods.Gaussian_with_sampler(noise_variance) for _ in range(Y_data.shape[-1])])
        
        # handle mean function initialization
        if mean_function is None:
            mean_function = gpnarx.mean_functions.MOMeanfunction([gpflow.mean_functions.Zero() for _ in range(Y_data.shape[-1])])
            
        # initialize parent class
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

    # type-ignore is because of changed method signature:
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        X, Y = self.data
        log_prob = tf.constant(0., dtype = default_float())
        for idx in range(self.num_latent_gps):
            likelihood = self.likelihood.likelihoods[idx]
            likelihood_variance = likelihood.variance
            kernel = self.kernel.kernels[idx]
            mean_function = self.mean_function.mean_functions[idx]
            X_data_1d, Y_data_1d = X, tf.expand_dims(Y[..., idx], axis = -1)
            
            K = kernel(X_data_1d)
            jitter = default_jitter() * tf.eye(K.shape[0], dtype=default_float())
            K += jitter
            ks = add_noise_cov(K, likelihood_variance)
            L = tf.linalg.cholesky(ks)
            m = mean_function(X)

            # [R,] log-likelihoods for each independent dimension of Y
            log_prob_1d = multivariate_normal(Y_data_1d, m, L)
            log_prob_1d = tf.reduce_sum(log_prob_1d)
            log_prob += log_prob_1d
        return log_prob

class MultivariateGPR_with_posterior(MultivariateGPR_deprecated):
    """
    Code based on gpflow.models.gpr.GPR @
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
    ) -> posteriors_multivariate.MultivariateGPRPosterior:
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

        return posteriors_multivariate.MultivariateGPRPosterior(
            kernel=self.kernel,
            data=self.data,
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
    
    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        X_data, _ = self.data
        mean, var = self.predict_f(X_data, full_cov = True)
        return mean, tf.transpose(var, perm = [2, 0, 1])
    
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
            training_loc, _ = self.data
            Kmm = kernel(training_loc)
            jitter = default_jitter() * tf.eye(Kmm.shape[0], dtype=default_float())
            Kmm += jitter
            L = tf.linalg.cholesky(Kmm)
            L_list.append(L[..., None])
            X_data_list.append(training_loc[..., None])
        L = tf.concat(L_list, axis = -1)
        X_data = tf.concat(X_data_list, axis = -1)
        return X_data, u_samples, L
    
class MultivariateGPR(MultivariateGPR_with_posterior):
    # subclassed to ensure __class__ == "GPR"
    pass

class MultivariateGPR_NARX(NARX, MultivariateGPR, GP_NARX):
    """
        Provides a NARX extension for the static GP model
    """
    def __init__(
        self,
        data: RegressionData,
        narx_params: list, 
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
        
        # initialize MultivariateGPR model
        MultivariateGPR.__init__(
            self, 
            data = (self.Z, self.Y), 
            *args, 
            **kwargs
        )
        
        # initialize GP-NARX
        GP_NARX.__init__(
            self
        )