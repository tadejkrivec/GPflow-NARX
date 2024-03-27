import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
# import progressbar as pb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import pickle
import os
import time
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
import gpflow 

def plot_model(pY, pYv, Y, output_list = None):
    t = range(pY.shape[0])
    for k in range(pY.shape[1]):
        plt.figure()
        plt.plot(t, Y[:, k], "kx")
        plt.gca().set_prop_cycle(None)
        plt.plot(t, pY[:, k], color = 'C0')
        top = pY[:, k] + 2.0 * pYv[:, k] ** 0.5
        bot = pY[:, k] - 2.0 * pYv[:, k] ** 0.5
        plt.fill_between(t, top, bot, alpha=0.3, color = 'C0')
        plt.xlabel("t")
        if output_list is None:
            plt.ylabel("y_" + str(k))
        else:
            plt.ylabel(output_list[k])

def plot_model_samples(samples, Y, output_list = None, num_samples = None, alpha = 0.2):
    t = range(samples.shape[1])
    if num_samples is None:
        num_samples = samples.shape[0]
    for k in range(samples.shape[-1]):
        plt.figure()
        plt.plot(t, Y[..., k], "kx")
        plt.gca().set_prop_cycle(None)
        plt.xlabel("t")
        num_samples_list = np.random.choice(np.array(range(samples.shape[0])), 
                                            size=num_samples, 
                                            replace=False)
        for j in num_samples_list:
            plt.plot(t, samples[j, :, k], color = 'C0', alpha = alpha)
        if output_list is None:
            plt.ylabel("y_" + str(k))
        else:
            plt.ylabel(output_list[k])
            
def optimize_batch_with_adam_and_natgrad(
    model,
    data,
    minibatch_size,
    log_mll = True, 
    compile_graph = False,
    max_iter = 10000, 
    adam_lr = 0.01,
    natgrad_lr = 0.1,
    libs = None
):
    if libs == None:
        assert False
    else:
        # unpack libs
        tf, gpflow = libs 
        
    @tf.function
    def optimization_step():
        gpflow.utilities.set_trainable(model.q_mu, False)
        gpflow.utilities.set_trainable(model.q_sqrt, False)
        
        adam_vars = model.trainable_variables
        natgrad_vars = [(model.q_mu, model.q_sqrt)]
        
        _ = natgrad_opt.minimize(loss_closure, natgrad_vars)
        _ = adam_opt.minimize(loss_closure, adam_vars)
        
    # initialite optimizer and  prepare data
    train_iter = iter(data.batch(minibatch_size))
    adam_opt = tf.optimizers.Adam(learning_rate = adam_lr)
    natgrad_opt = gpflow.optimizers.NaturalGradient(gamma = natgrad_lr)
    
    loss_closure = model.training_loss_closure(train_iter, compile = compile_graph)
    
    logger = []
    for step in range(max_iter):
        optimization_step()
            
        if log_mll is True:
            nmll = loss_closure().numpy()
            logger.append(nmll)         
    return logger

def optimize_model_with_adam(
    model, 
    log_mll = True, 
    compile_graph = False,
    max_iter = 10000, 
    learning_rate = 0.01, 
    libs = None
):   
    if libs == None:
        assert False
    else:
        # unpack libs
        tf, gpflow = libs 
        
    @tf.function
    def optimization_step():
        _ = opt.minimize(loss_closure, model.trainable_variables) 
        
    # initialite optimizer
    opt = tf.optimizers.Adam(learning_rate = learning_rate)
    loss_closure = model.training_loss_closure(compile = compile_graph)
    
    logger = []
    for step in range(max_iter):
        optimization_step()
            
        if log_mll is True:
            nmll = loss_closure().numpy()
            logger.append(nmll)         
    return logger

def optimize_model_with_scipy(model,
    log_mll = True,
    compile_graph = False,
    max_iter = 10000,
    libs = None,
    data = None
):
    if libs == None:
        assert False
    else:
        # unpack libs
        tf, gpflow = libs 
        
    def append_log(loss_closure):
        def _append(*args, **kwargs):
            logger.append(loss_closure().numpy())
        return _append
        
    opt = gpflow.optimizers.Scipy()
    if data is None:
        loss_closure = model.training_loss_closure(compile = compile_graph)
    else:
        loss_closure = model.training_loss_closure(data = data, compile = compile_graph)
    
    if log_mll is True:
        step_callback = append_log(loss_closure)
    else:
        step_callback = None
        
    logger = []       
    _ = opt.minimize(
        loss_closure,
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options=dict(maxiter=max_iter),
        compile=compile_graph,
        step_callback = step_callback
    )
    return logger

def optimize_batch_with_adam(
    model,
    data,
    minibatch_size,
    log_mll = True, 
    compile_graph = False,
    max_iter = 10000, 
    learning_rate = 0.01,
    libs = None
):
    if libs == None:
        assert False
    else:
        # unpack libs
        tf, gpflow = libs 
        
    @tf.function
    def optimization_step():
        _ = opt.minimize(loss_closure, model.trainable_variables)  
        
    # initialite optimizer and  prepare data
    train_iter = iter(data.batch(minibatch_size))
    opt = tf.optimizers.Adam(learning_rate = learning_rate)
    loss_closure = model.training_loss_closure(train_iter, compile = compile_graph)
    
    logger = []
    for step in range(max_iter):
        optimization_step()
            
        if log_mll is True:
            nmll = loss_closure().numpy()
            logger.append(nmll)         
    return logger