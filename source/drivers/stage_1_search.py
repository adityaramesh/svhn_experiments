from subprocess import Popen

sgu_learning_rate_list     = [0.5, 0.4, 0.3, 0.2, 0.1]
rmsprop_learning_rate_list = [0.003, 0.0025, 0.002, 0.0015, 0.001]
learning_rate_decay_list   = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
decay_list                 = [0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
momentum_list              = [0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
epsilon_list               = [1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
lambda_list                = [1 - 1e-8, 1 - 1e-9, 1 - 1e-10, 1 - 1e-11, 1 - 1e-12]
beta_1_list                = [0, 0.9]
beta_2_list                = [0.99, 0.999, 0.9999]

class ResourceManager:
    def __init__(self):
        pass

    def submit(self):
        pass

# TODO: SGU
# TODO: AdaDelta
# TODO: RMSProp
# TODO: Adam
