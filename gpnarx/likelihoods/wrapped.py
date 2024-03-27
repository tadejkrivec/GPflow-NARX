from gpflow.likelihoods.scalar_continuous import Gaussian
import tensorflow_probability as tfp

class Gaussian_with_sampler(Gaussian):
    def sample(self, f, num_samples):
        return tfp.distributions.Normal(loc = f, scale = self.variance**0.5).sample(num_samples)
    
    def prob(self, y_predicted, y_true):
        return tfp.distributions.Normal(loc = y_true, scale = self.variance**0.5).prob(y_predicted)