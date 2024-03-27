import abc
from typing import Optional, Tuple, Type, Union, Callable, List

import tensorflow as tf

from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.posteriors import PrecomputeCacheType
from gpflow.models.util import data_input_to_tensor
from gpflow import default_float

from .. import posteriors_multivariate, posteriors_multivariate_pimc
from ..conditionals import Iterative_dynamic_sampling_conditional_with_precompute
        
class NARX_parameters:
    '''
        Class that stores the parameters of the NARX model
    '''
    def __init__(
        self, 
        nq: Union[int, List[int]], 
        na: Union[int, List[int]], 
        nk: Optional[int] = 0, 
        nb: Optional[int] = 0, 
        lambda_list: List[Callable] = None,
    ):
        
        # initializes exogenous parameters
        self.nk = nk
        self.nb = nb
        
        # initializes ar parameters
        assert type(nq) == type(na)
        if type(nq) == int:
            self.nq = [nq]
            self.na = [na]
        else:
            self.nq = nq
            self.na = na
        
        # makes sure lambda list is initialized
        if lambda_list is None:
            self.lambda_list = [lambda x: x]
        else:
            self.lambda_list = lambda_list
        
        # calculates max lag
        self.max_lag = max(nk+nb, max([nq + na for nq, na in zip(self.nq, self.na)])) - 1
        
        # sanity checks the parameters
        self.check_params()
        
    def get_input_dims(
        self, 
        Y: tf.Tensor, 
        U: Optional[tf.Tensor] = None
    ):
        '''
            Gets the dimension parameters in NARX inputs
        '''
        
        if U is not None:
            xdim = self.nb*U.shape[1]
        else:
            xdim = 0
        ydim = sum([f(Y).shape[1]*na for na, f in zip(self.na, self.lambda_list)])
        ndim = xdim + ydim
        return xdim, ydim, ndim  
    
    def check_params(self):
        '''
            Sanity check of the NARX parameters
        '''
        
        assert type(self.nq) == type(self.na) == list
        assert type(self.nk) == type(self.nb) == int
        assert len(self.nq) == len(self.na) == len(self.lambda_list)
        assert type(self.lambda_list) == list
        assert all([x > 0 for x in self.nq])
        assert all([x > 0 for x in self.na])
        assert self.nk >= 0
        assert self.nb >= 0 
        
class NARX():
    '''
        Class with methods that can build the input/output pairs of a NARX structure
    '''
        
    def __init__(
        self,
        Y: tf.Tensor,
        narx_params: NARX_parameters,
        U: Optional[tf.Tensor] = None
    ):
        
        # adds narx params to the object
        self.narx_params = narx_params
        
        # gets the maximal lag value
        if U is not None:
            self.U_ncol, self.Y_ncol  = U.shape[1], Y.shape[1]
            assert self.narx_params.nb > 0
        else:
            self.U_ncol, self.Y_ncol = None, Y.shape[1]
            assert self.narx_params.nk == self.narx_params.nb == 0
            
        # gets the input dims 
        xdim, ydim, ndim  = self.narx_params.get_input_dims(Y, U)
        self.xdim, self.ydim, self.ndim = xdim, ydim, ndim
        
        # constructs narx model
        Z, Y = self.construct_narx(U, Y)
        self.Z, self.Y = Z[0, ...], Y[0, ...]
    
    def get_lag(
        self, 
        x: tf.Tensor, 
        delay_list: List[int], 
        lag_list: List[int], 
        lambda_list: List[Callable]
    ):
        '''
            Constructs the augmented regressior with the lags specified by the delay_list, lag_list, and lambda_list
        '''
        
        # constructs all parameter combinations
        x_lagged_zip = []
        for delay, lag, f in zip(delay_list, lag_list, lambda_list):
            for lag_value in range(delay, delay+lag):
                x_lagged_zip.append((lag_value, f))
        
        # constructs the lagged matrix
        x_lagged_list = []
        for lag_value, f in x_lagged_zip:
            x_lagged_list.append(tf.roll(f(x), shift = lag_value, axis = 1))
        x_lagged = tf.concat(x_lagged_list, axis = 2)
        
        # cuts the lags at the start 
        x_lagged, x = x_lagged[:, self.narx_params.max_lag:, :], x[:, self.narx_params.max_lag:, :]
        
        # returns lagged matrix and original vector/matrix
        return x_lagged, x
        
    def construct_narx(
        self, 
        U: tf.Tensor, 
        Y: tf.Tensor
    ):
        '''
            Contructs the NARX model input/output pair
        '''
        
        # expands the input data to handle batch dimension
        if (U is not None): 
            if (U.shape.rank == 2) & (Y.shape.rank == 2):
                U, Y = U[None, ...], Y[None, ...]
        elif (U is None) & (Y.shape.rank == 2):
            Y = Y[None, ...]

        # gets Y_lag
        Y_lag, Y = self.get_lag(Y, self.narx_params.nq, self.narx_params.na, self.narx_params.lambda_list)
        
        # checks if model is AR
        if U is not None:
            # gets U lag
            U_lag, _ = self.get_lag(U, [self.narx_params.nk], [self.narx_params.nb], [lambda x: x])
            
            # constructs input data
            Z = tf.concat([U_lag, Y_lag], axis = 2)
            
        else:
            # gets U lag
            U_lag = None
            
            # constructs input data
            Z = Y_lag   
            
        # removes nan rows
        mask = ~(tf.reduce_any(tf.math.is_nan(Z), axis = [0, 2]) | tf.reduce_any(tf.math.is_nan(Y), axis = [0, 2]))
        Z, Y = tf.boolean_mask(Z, mask, axis = 1),  tf.boolean_mask(Y, mask, axis = 1)
            
        # returns the input regression and the corresponding output
        return Z, Y
    
class GP_NARX(): 
    '''
        Class that adds the dynamic functionality to the NARX model:
            - Prediction methods
            - Simulation methods
    ''' 
    def __init__(
        self,
    ):
        pass
    
    @abc.abstractmethod
    def posterior(
        self,
        precompute_cache: PrecomputeCacheType = PrecomputeCacheType.TENSOR,
    ):
        raise NotImplementedError
    
    @abc.abstractmethod
    def predict_f(
        self, 
        Xnew: InputData, 
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        full_cov: bool = False, 
        full_output_cov: bool = False
    ) -> MeanAndVariance:
        raise NotImplementedError
        
    @abc.abstractmethod    
    def predict_f_samples(
        self,
        Xnew: InputData,
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        num_samples: Optional[int] = None,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        raise NotImplementedError
    
    def predict_latent(
        self, 
        Unew: InputData,
        Ynew: InputData,
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        full_cov: bool = False, 
        full_output_cov: bool = False
    ) -> MeanAndVariance:
        '''
            Constructs the NARX model from static data and predicts the latent function values
        '''
        
        # converts the data to TF constants
        Unew, Ynew  = data_input_to_tensor(Unew), data_input_to_tensor(Ynew)
                
        # constructs input/output data as NARX
        Z, _ = self.construct_narx(Unew, Ynew)
        Z = Z[0, ...]
        
        if observed_data is not None:
            Z_observed, Y_observed  = data_input_to_tensor(observed_data[0]), data_input_to_tensor(observed_data[1])
            
            Z_observed, Y_observed = self.construct_narx(
                Z_observed,
                Y_observed
            )
            Z_observed, Y_observed = Z_observed[0, ...], Y_observed[0, ...] 
            observed_data = (Z_observed, Y_observed)
            
        # predicts with cached data
        return self.posterior(precompute_cache = posteriors_multivariate.PrecomputeCacheType.TENSOR).predict_f(
            Z, 
            observed_data = observed_data,
            observed_latent = observed_latent,
            full_cov = full_cov, 
            full_output_cov = full_output_cov
        )       

    def predict_latent_samples(
        self,
        Unew: InputData,
        Ynew: InputData, 
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        num_samples: Optional[int] = None,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        '''
            Constructs the NARX model from static data and samples the latent function
        '''
        
        # converts the data to TF constants
        Unew, Ynew  = data_input_to_tensor(Unew), data_input_to_tensor(Ynew)
                
        # constructs input/output data as NARX
        Z, _ = self.construct_narx(Unew, Ynew)
        Z = Z[0, ...]

        if observed_data is not None:
            Z_observed, Y_observed  = data_input_to_tensor(observed_data[0]), data_input_to_tensor(observed_data[1])
            
            Z_observed, Y_observed = self.construct_narx(
                Z_observed,
                Y_observed
            )
            Z_observed, Y_observed = Z_observed[0, ...], Y_observed[0, ...]
            observed_data = (Z_observed, Y_observed)
        
        # samples with cached data
        return self.posterior(precompute_cache = posteriors_multivariate.PrecomputeCacheType.TENSOR).predict_f_samples(
            Z, 
            observed_data = observed_data,
            observed_latent = observed_latent,
            full_cov = full_cov, 
            full_output_cov = full_output_cov
        )         

    def predict_noisy(
        self, 
        Unew: InputData,
        Ynew: InputData,
        observed_data: RegressionData = None,
        observed_latent: bool = True,
        full_cov: bool = False, 
        full_output_cov: bool = False
    ) -> MeanAndVariance:
        '''
            Constructs the NARX model from static data and predicts the noisy function values
        '''
        
        # converts the data to TF constants
        Unew, Ynew  = data_input_to_tensor(Unew), data_input_to_tensor(Ynew)
        
        # sanity check for stuff that does not work
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False"
                " and full_output_cov=False"
            )
            
        if observed_data is not None:
            Z_observed, Y_observed  = data_input_to_tensor(observed_data[0]), data_input_to_tensor(observed_data[1])
            
            Z_observed, Y_observed = self.construct_narx(
                Z_observed,
                Y_observed
            )
            Z_observed, Y_observed = Z_observed[0, ...], Y_observed[0, ...]
            observed_data = (Z_observed, Y_observed)
            
        # predicts the latent function
        f_mean, f_var = self.predict_latent(
            Unew, 
            Ynew, 
            observed_data = observed_data,
            observed_latent = observed_latent,
            full_cov=full_cov, 
            full_output_cov=full_output_cov)
        
        # adds the likelihood noise
        y_mean, y_var = self.likelihood.predict_mean_and_var(f_mean, f_var)
        
        # returns noisy prediction
        return y_mean, y_var 

    def predict_log_density(
        self, 
        Unew: InputData,
        Ynew: InputData,
        full_cov: bool = False, 
        full_output_cov: bool = False
    ) -> tf.Tensor:
        '''
            Predicts the log density
        '''
        
        # converts the data to TF constants
        Unew, Ynew  = data_input_to_tensor(Unew), data_input_to_tensor(Ynew)
        
        # sanity check for stuff that does not work
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_log_density method currently supports only the argument values"
                " full_cov=False and full_output_cov=False"
            )
            
        # predicts the latent function
        f_mean, f_var = self.predict_latent(Unew, Ynew, full_cov=full_cov, full_output_cov=full_output_cov)
        
        # returns the log density
        return self.likelihood.predict_log_density(f_mean, f_var, Y)

    
    def simulate_latent_montecarlo(
        self, 
        y_initial: tf.Tensor,
        U: Optional[tf.Tensor] = None, 
        num_steps: Optional[int] =  None,
        full_cov: Optional[bool] = False, 
        propagate_error: Optional[bool] = False,
        num_samples: Optional[int] = 100
):
        pass
    
    def simulate_latent_montecarlo_independent(
        self, 
        y_initial: tf.Tensor,
        U: Optional[tf.Tensor] = None,
        propagate_error: Optional[bool] = False,
        num_steps: Optional[int] =  None,
        num_samples: Optional[int] = 100,
        compile: bool = True
    ):
        '''
            Simulation where the consequent latent functions values are independent:
            p(f_next | f_prev, y) = p(f_next | y)
        '''
        
        # converts to TF constant
        y_initial  = data_input_to_tensor(y_initial)
        if U is not None:
            U = data_input_to_tensor(U)
        
        # Either provide U or num_steps
        assert (U == None) | (num_steps == None)
        assert y_initial.shape[0] == self.narx_params.max_lag

        # gets the number of steps
        if U is not None:
            num_steps = U.shape[0] - self.narx_params.max_lag

        # expands the data so the prediction can be obtained concurrently
        dummy_fill = tf.zeros(shape = (num_samples, 1, y_initial.shape[-1]), dtype = default_float())
        y_buffer = tf.repeat(y_initial[None, ...], repeats = num_samples, axis = 0)
        y_buffer = tf.concat([y_buffer, dummy_fill], axis = 1) 
        U = tf.repeat(U[None, ...], repeats = num_samples, axis = 0)

        # initilizes the posterior object that allows caching and compile the time consuming functions if needed
        posterior = self.posterior(precompute_cache = posteriors_multivariate.PrecomputeCacheType.TENSOR)
        if compile:
            predict_f_samples = tf.function(posterior.predict_f_samples)
            construct_narx = tf.function(self.construct_narx)
            if propagate_error:
                likelihood_sample = tf.function(self.likelihood._sample)
        else:
            predict_f_samples = posterior.predict_f_samples
            construct_narx = self.construct_narx
            if propagate_error:
                likelihood_sample = self.likelihood._sample

        # initializes tensorarray to collect data in a loop
        output_samples = tf.TensorArray(dtype = default_float(), size = num_steps)

        # iterates through the steps and predict sequentially
        for step in range(num_steps):
            # gets the input regressor for the time step considered
            U_iter = U[:, step:step + self.narx_params.max_lag + 1, :]
            Z_iter, _ = construct_narx(U = U_iter, Y = y_buffer)

            # runs the independent prediction and sample from it
            latent_samples = predict_f_samples(
                Z_iter[:, 0, :],
                num_samples = 1,
                full_cov = False
            )
            latent_samples = tf.transpose(latent_samples, perm = [1, 0, 2])

            # saves results for the iteration
            output_samples = output_samples.write(step, latent_samples[:, 0, :])
            
            # samples from the likelihood if the error gets propagated
            # the likelihood needs to define a _sample method
            if propagate_error:
                assert hasattr(self.likelihood, '_sample') and callable(getattr(self.likelihood, '_sample'))
                noisy_samples = likelihood_sample(F = latent_samples, num_samples = 1)
                noisy_samples = noisy_samples[0, ...]
            else:
                noisy_samples = latent_samples

            # update the variable with new samples
            y_buffer = tf.concat([y_buffer[:, 1:-1, :], noisy_samples, dummy_fill], axis = 1)

        # permutes the results to lead with samples
        output_samples = output_samples.stack()
        output_samples = tf.transpose(output_samples, perm=[1, 0, 2])
        
        # returns independent MC samples
        return output_samples


    def simulate_naive_independent(
        self, 
        y_initial: tf.Tensor,
        U: Optional[tf.Tensor] = None,
        num_steps: Optional[int] =  None,
        compile: bool = True
    ):
        '''
            Simulation where the consequent latent functions values are independent:
            p(f_next | f_prev, y) = p(f_next | y),
            and mean is propagated instead of a latent sample:
            f_sampled = f_mean
        '''
        # converts to TF constant
        y_initial  = data_input_to_tensor(y_initial)
        if U is not None:
            U = data_input_to_tensor(U)
            
        # Either provide U or num_steps
        assert (U == None) | (num_steps == None)
        assert y_initial.shape[0] == self.narx_params.max_lag

        # gets the number of steps
        if U is not None:
            num_steps = U.shape[0] - self.narx_params.max_lag

        # initializes the buffer
        dummy_fill = tf.zeros(shape = (1, y_initial.shape[1]), dtype = default_float())
        y_buffer = tf.concat([y_initial, dummy_fill], axis = 0) 

        # initilizes the posterior object that allows caching and compile time consuming functions if needed
        posterior = self.posterior(precompute_cache = posteriors_multivariate.PrecomputeCacheType.TENSOR)
        if compile:
            predict_f = tf.function(posterior.predict_f)
            construct_narx = tf.function(self.construct_narx)
        else:
            predict_f = posterior.predict_f
            construct_narx = self.construct_narx
                
        # initializes tensorarray to collect data in a loop
        output_mean = tf.TensorArray(dtype = default_float(), size = num_steps)
        output_var = tf.TensorArray(dtype = default_float(), size = num_steps)

        # iterates through the steps and predicts sequentially
        for step in range(num_steps):

            # gets the input regressor for the time step considered
            U_iter = U[step:step + self.narx_params.max_lag + 1, :]
            Z_iter, _ = construct_narx(U = U_iter, Y = y_buffer)
  
            # runs the independent prediction
            mean, var = predict_f(
                Z_iter[0, ...],
                full_cov = False
            )
  
            # saves results for the iteration
            output_mean = output_mean.write(step, mean)
            output_var = output_var.write(step, var)

            # updates the variable with new mean
            y_buffer = tf.concat([y_buffer[1:-1], mean, dummy_fill], axis = 0)

        # permutes the results to lead with samples
        output_mean = output_mean.stack()
        output_var = output_var.stack()
        output_mean = output_mean[:, 0, :]
        output_var = output_var[:, 0, :]
        
        # returns the mean and the variance
        return output_mean, output_var
    
    def simulate_latent_montecarlo_correlated(
        self, 
        y_initial: tf.Tensor,
        U: Optional[tf.Tensor] = None,
        propagate_error: Optional[bool] = False,
        num_steps: Optional[int] =  None,
        num_samples: Optional[int] = 100,
        std_threshold = tf.constant(0., dtype = default_float()),
        compile: bool = True
    ):
        '''
            Simulation where the consequent latent functions values are independent:
            p(f_next | f_prev, y) = p(f_next | y)
        '''
        
        # converts to TF constant
        y_initial  = data_input_to_tensor(y_initial)
        if U is not None:
            U = data_input_to_tensor(U)
        
        # Either provide U or num_steps
        assert (U == None) | (num_steps == None)
        assert y_initial.shape[0] == self.narx_params.max_lag

        # gets the number of steps
        if U is not None:
            num_steps = U.shape[0] - self.narx_params.max_lag
        
        # expands the data so the prediction can be obtained concurrently
        dummy_fill = tf.zeros(shape = (num_samples, 1, y_initial.shape[-1]), dtype = default_float())
        y_buffer = tf.repeat(y_initial[None, ...], repeats = num_samples, axis = 0)
        y_buffer = tf.concat([y_buffer, dummy_fill], axis = 1) 
        U = tf.repeat(U[None, ...], repeats = num_samples, axis = 0)

        # initilizes the cache, i.e., samples from the inducing point posterior
        Z_loc, f_samples, L = self.predict_qu_samples(num_samples = num_samples)
        L = tf.repeat(L[None, ...], repeats = num_samples, axis = 0)
        Z_loc = tf.repeat(Z_loc[None, ...], repeats = num_samples, axis = 0)
        L, X, f = L, Z_loc, f_samples
        OUTPUT_DIM = f.shape[-1]
        INPUT_DIM = Z_loc.shape[-2]
        shapes = (num_samples, OUTPUT_DIM, INPUT_DIM)

        # initialize interative gaussian conditional
        iterative_sampling_conditional = Iterative_dynamic_sampling_conditional_with_precompute(
            kernel = self.kernel,
            mean_function = self.mean_function,
            likelihood = self.likelihood,
            shapes = shapes,
            compile = compile,
            std_threshold = std_threshold
        )
    
        # initializes tensorarray to collect data in a loop
        output_samples = tf.TensorArray(dtype = default_float(), size = num_steps)

        # iterates through the steps and predict sequentially
        for step in range(num_steps):
            #print('N step={0}'.format(step))
            
            # gets the input regressor for the time step considered
            U_iter = U[:, step:step + self.narx_params.max_lag + 1, :]
            Z_iter, _ = self.construct_narx(U = U_iter, Y = y_buffer)
  
            # runs the correlated prediction and sample from it
            latent_samples, L, X, f  = iterative_sampling_conditional.predict_f_sample(Z_iter, L, X, f)
            
            #print('N latent_samples={0}'.format(f.shape[1]))
            # saves results for the iteration
            output_samples = output_samples.write(step, latent_samples)

            # samples from the likelihood if the error gets propagated
            # the likelihood needs to define a _sample method
            if propagate_error:
                assert hasattr(self.likelihood, '_sample') and callable(getattr(self.likelihood, '_sample'))
                noisy_samples = self.likelihood._sample(F = latent_samples, num_samples = 1)
                noisy_samples = noisy_samples[0, ...]
            else:
                noisy_samples = latent_samples

            # update the variable with new samples
            y_buffer = tf.concat([y_buffer[:, 1:-1, :], noisy_samples[:, None, :], dummy_fill], axis = 1)

        # permutes the results to lead with samples
        output_samples = output_samples.stack()
        output_samples = tf.transpose(output_samples, perm=[1, 0, 2])
        
        # returns independent MC samples
        return output_samples
    
    
    def simulate_latent_montecarlo_correlated_retainedpoints(
        self, 
        y_initial: tf.Tensor,
        U: Optional[tf.Tensor] = None,
        propagate_error: Optional[bool] = False,
        num_steps: Optional[int] =  None,
        num_samples: Optional[int] = 100,
        std_threshold = tf.constant(0., dtype = default_float()),
        compile: bool = True
    ):
        '''
            Simulation where the consequent latent functions values are independent:
            p(f_next | f_prev, y) = p(f_next | y)
        '''
        
        # converts to TF constant
        y_initial  = data_input_to_tensor(y_initial)
        if U is not None:
            U = data_input_to_tensor(U)
        
        # Either provide U or num_steps
        assert (U == None) | (num_steps == None)
        assert y_initial.shape[0] == self.narx_params.max_lag

        # gets the number of steps
        if U is not None:
            num_steps = U.shape[0] - self.narx_params.max_lag
        
        # expands the data so the prediction can be obtained concurrently
        dummy_fill = tf.zeros(shape = (num_samples, 1, y_initial.shape[-1]), dtype = default_float())
        y_buffer = tf.repeat(y_initial[None, ...], repeats = num_samples, axis = 0)
        y_buffer = tf.concat([y_buffer, dummy_fill], axis = 1) 
        U = tf.repeat(U[None, ...], repeats = num_samples, axis = 0)

        # initilizes the cache, i.e., samples from the inducing point posterior
        Z_loc, f_samples, L = self.predict_qu_samples(num_samples = num_samples)
        L = tf.repeat(L[None, ...], repeats = num_samples, axis = 0)
        Z_loc = tf.repeat(Z_loc[None, ...], repeats = num_samples, axis = 0)
        L, X, f = L, Z_loc, f_samples
        OUTPUT_DIM = f.shape[-1]
        INPUT_DIM = Z_loc.shape[-2]
        shapes = (num_samples, OUTPUT_DIM, INPUT_DIM)

        # initialize interative gaussian conditional
        iterative_sampling_conditional = Iterative_dynamic_sampling_conditional_with_precompute(
            kernel = self.kernel,
            mean_function = self.mean_function,
            likelihood = self.likelihood,
            shapes = shapes,
            compile = compile,
            std_threshold = std_threshold
        )
    
        # initializes tensorarray to collect data in a loop
        output_samples = tf.TensorArray(dtype = default_float(), size = num_steps)
        retained_points_list = []
        
        # iterates through the steps and predict sequentially
        for step in range(num_steps):
            #print('N step={0}'.format(step))
            
            # gets the input regressor for the time step considered
            U_iter = U[:, step:step + self.narx_params.max_lag + 1, :]
            Z_iter, _ = self.construct_narx(U = U_iter, Y = y_buffer)
  
            # runs the correlated prediction and sample from it
            latent_samples, L, X, f  = iterative_sampling_conditional.predict_f_sample(Z_iter, L, X, f)
            
            #print('N latent_samples={0}'.format(f.shape[1]))
            retained_points_iter = f.shape[1]
            retained_points_list.append(retained_points_iter)
            
            # saves results for the iteration
            output_samples = output_samples.write(step, latent_samples)

            # samples from the likelihood if the error gets propagated
            # the likelihood needs to define a _sample method
            if propagate_error:
                assert hasattr(self.likelihood, '_sample') and callable(getattr(self.likelihood, '_sample'))
                noisy_samples = self.likelihood._sample(F = latent_samples, num_samples = 1)
                noisy_samples = noisy_samples[0, ...]
            else:
                noisy_samples = latent_samples

            # update the variable with new samples
            y_buffer = tf.concat([y_buffer[:, 1:-1, :], noisy_samples[:, None, :], dummy_fill], axis = 1)

        # permutes the results to lead with samples
        output_samples = output_samples.stack()
        output_samples = tf.transpose(output_samples, perm=[1, 0, 2])
        
        # returns independent MC samples
        return output_samples, retained_points_list

    def simulate_latent_montecarlo_pseudoindependent(
        self, 
        y_initial: tf.Tensor,
        U: Optional[tf.Tensor] = None,
        propagate_error: Optional[bool] = False,
        num_steps: Optional[int] =  None,
        num_samples: Optional[int] = 100,
        compile: bool = True
    ):
        '''
            Simulation where the consequent latent functions values are independent:
            p(f_next | f_prev, y) = p(f_next | y)
        '''
        
        # converts to TF constant
        y_initial  = data_input_to_tensor(y_initial)
        if U is not None:
            U = data_input_to_tensor(U)
        
        # Either provide U or num_steps
        assert (U == None) | (num_steps == None)
        assert y_initial.shape[0] == self.narx_params.max_lag

        # gets the number of steps
        if U is not None:
            num_steps = U.shape[0] - self.narx_params.max_lag

        # expands the data so the prediction can be obtained concurrently
        dummy_fill = tf.zeros(shape = (num_samples, 1, y_initial.shape[-1]), dtype = default_float())
        y_buffer = tf.repeat(y_initial[None, ...], repeats = num_samples, axis = 0)
        y_buffer = tf.concat([y_buffer, dummy_fill], axis = 1) 
        U = tf.repeat(U[None, ...], repeats = num_samples, axis = 0)
        
        # initilizes the cache, i.e., samples from the inducing point posterior
        Z_loc, f_samples, L = self.predict_qu_samples(num_samples = num_samples)
        
        # initilizes the posterior object that allows caching and compile the time consuming functions if needed
        posterior = posteriors_multivariate_pimc.MultivariatePIMCPosterior(
            kernel=self.kernel,
            mean_function=self.mean_function,
            data = (Z_loc, f_samples),
            L = L,
            precompute_cache= posteriors_multivariate_pimc.PrecomputeCacheType.TENSOR,
        )
        
        if compile:
            predict_f_samples = tf.function(posterior.predict_f_samples)
            construct_narx = tf.function(self.construct_narx)
            if propagate_error:
                likelihood_sample = tf.function(self.likelihood._sample)
        else:
            predict_f_samples = posterior.predict_f_samples
            construct_narx = self.construct_narx
            if propagate_error:
                likelihood_sample = self.likelihood._sample

        # initializes tensorarray to collect data in a loop
        output_samples = tf.TensorArray(dtype = default_float(), size = num_steps)

        # iterates through the steps and predict sequentially
        for step in range(num_steps):
            # gets the input regressor for the time step considered
            U_iter = U[:, step:step + self.narx_params.max_lag + 1, :]
            Z_iter, _ = construct_narx(U = U_iter, Y = y_buffer)

            # runs the independent prediction and sample from it
            latent_samples = predict_f_samples(
                Z_iter[:, 0, :],
                num_samples = 1,
                full_cov = False
            )
            latent_samples = tf.transpose(latent_samples, perm = [1, 0, 2])

            # saves results for the iteration
            output_samples = output_samples.write(step, latent_samples[:, 0, :])
            
            # samples from the likelihood if the error gets propagated
            # the likelihood needs to define a _sample method
            if propagate_error:
                assert hasattr(self.likelihood, '_sample') and callable(getattr(self.likelihood, '_sample'))
                noisy_samples = likelihood_sample(F = latent_samples, num_samples = 1)
                noisy_samples = noisy_samples[0, ...]
            else:
                noisy_samples = latent_samples

            # update the variable with new samples
            y_buffer = tf.concat([y_buffer[:, 1:-1, :], noisy_samples, dummy_fill], axis = 1)

        # permutes the results to lead with samples
        output_samples = output_samples.stack()
        output_samples = tf.transpose(output_samples, perm=[1, 0, 2])
        
        # returns independent MC samples
        return output_samples