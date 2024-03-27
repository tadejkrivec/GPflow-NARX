import tensorflow as tf

from gpflow.mean_functions import MeanFunction

class MOMeanfunction(MeanFunction):
    def __init__(self, mean_function_list, **kwargs):
        for m in mean_function_list:
            assert isinstance(m, MeanFunction)
        self.mean_functions = mean_function_list

    def __call__(self, X):
        call_list = []
        for m in self.mean_functions:
            call_list.append(m(X))
        return tf.concat(call_list, axis = 1)