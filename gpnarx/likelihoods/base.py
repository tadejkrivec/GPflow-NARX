import abc
import warnings
from typing import Any, Callable, Iterable, Optional, Sequence

import tensorflow as tf

from gpflow.likelihoods import ScalarLikelihood

class MOLikelihood(ScalarLikelihood):
    def __init__(self, likelihood_list, **kwargs):
        """
        In this likelihood, we assume at extra column of Y, which contains
        integers that specify a likelihood from the list of likelihoods.
        """
        super().__init__(**kwargs)
        for l in likelihood_list:
            assert isinstance(l, ScalarLikelihood)
        self.likelihoods = likelihood_list

    def _scalar_log_prob(self, F, Y):
        log_prob_list = [lik._scalar_log_prob(F[:, i][:, None], Y[:, i][:, None]) for i, lik in enumerate(self.likelihoods)]
        log_prob = tf.concat(log_prob_list, axis=1)
        return log_prob

    def _predict_log_density(self, Fmu, Fvar, Y):
        log_density_list = [lik.predict_log_density(Fmu[:, i][:, None], Fvar[:, i][:, None], Y[:, i][:, None]) for i, lik in enumerate(self.likelihoods)]
        log_density = tf.concat(log_density_list, axis=1)
        return log_density

    def _variational_expectations(self, Fmu, Fvar, Y):
        var_exp_list = [lik.variational_expectations(Fmu[:, i][:, None], Fvar[:, i][:, None], Y[:, i][:, None]) for i, lik in enumerate(self.likelihoods)]
        var_exp_list = [x[:, None] for x in var_exp_list]
        var_exp = tf.concat(var_exp_list, axis=1)
        return tf.reduce_sum(var_exp, axis = 1)
    
    def _predict_mean_and_var(self, Fmu, Fvar):
        mvs = [lik.predict_mean_and_var(Fmu[..., i][..., None], Fvar[..., i][..., None]) for i, lik in enumerate(self.likelihoods)]
        
        mu_list, var_list = zip(*mvs)
        mu = tf.concat(mu_list, axis=-1)
        var = tf.concat(var_list, axis=-1)
        return mu, var

    def _conditional_mean(self, F):
        raise NotImplementedError

    def _conditional_variance(self, F):
        raise NotImplementedError
        
    def _sample(self, F, num_samples):
        samples_list = [lik.sample(f = F[..., i][..., None], num_samples = num_samples) for i, lik in enumerate(self.likelihoods)]
        
        samples = tf.concat(samples_list, axis=-1)
        return samples
        