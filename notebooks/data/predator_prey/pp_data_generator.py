import numpy as np
class Generator():
    def __init__(self, 
                 train_steps = 500, 
                 test_steps = 500,
                 train_init = [0.3, 0.8],
                 test_init = [0.268, 0.4],
                 noise_levels = [0.01, 0.01]):
        
        self.train_steps = train_steps
        self.test_steps = test_steps
        self.train_init = train_init
        self.test_init = test_init
        self.noise_levels = noise_levels
        
    def get_predator_prey_data(self):
        # create predator-prey system functions
        fp1 = lambda x1, x2, u1: x1*np.exp(1 - 0.4*x1 - ((2 + 1.2*u1)*x2)/(1 + x1**2)) 
        fp2 = lambda x1, x2, u1, u2: x2*np.exp(1 + 0.5*u1 - ((1.5 - u2)*x2)/x1)
        u1 = lambda t: np.cos(0.02 * np.pi * t)
        u2 = lambda t: np.sin(0.02 * np.pi * t)

        # initialize the system
        steps = self.train_steps
        xt1, xt2 = np.zeros((steps, 1)), np.zeros((steps, 1))
        xt1[0], xt2[0] = self.train_init[0], self.train_init[1]

        # simulate the system
        for t in range(steps - 1):
            # calculate and update the states
            xt1[t+1] = fp1(xt1[t], xt2[t], u1(t))
            xt2[t+1] = fp2(xt1[t], xt2[t], u1(t), u2(t))

        # get train 
        t = np.linspace(0, steps-1, steps)
        subsample = 1
        u_train = np.concatenate([u1(t)[::subsample, None], u2(t)[::subsample, None]], axis = 1)
        x_train = np.concatenate([xt1[::subsample], xt2[::subsample]], axis = 1)

        # insert dependent noise
        mean = np.array([0., 0.])
        cov = np.array([[self.noise_levels[0], 0.], [0., self.noise_levels[1]]])
        y_train = x_train + np.random.multivariate_normal(mean, cov, x_train.shape[0])

        # initialize the system for test
        steps = self.test_steps
        xt1, xt2 = np.zeros((steps, 1)), np.zeros((steps, 1))
        xt1[0], xt2[0] = self.test_init[0], self.test_init[1]

        # simulate the system
        for t in range(steps - 1):
            # calculate and update the states
            xt1[t+1] = fp1(xt1[t], xt2[t], u1(t))
            xt2[t+1] = fp2(xt1[t], xt2[t], u1(t), u2(t))

        # get test 
        t = np.linspace(0, steps-1, steps)
        u_test = np.concatenate([u1(t)[:, None], u2(t)[:, None]], axis = 1)
        x_test = np.concatenate([xt1, xt2], axis = 1)

        # # # standardize 
        # x_test, _, _ = standardize(x_test)
        y_test = x_test + np.random.multivariate_normal(mean, cov, x_test.shape[0])

        # remove transients
        y_train = y_train[100:, :]
        x_train = x_train[100:, :]
        u_train = u_train[100:, :]
        x_test = x_test[100:, :]
        y_test = y_test[100:, :]
        u_test = u_test[100:, :]

        return y_train, y_test, x_train, x_test, u_train, u_test
        