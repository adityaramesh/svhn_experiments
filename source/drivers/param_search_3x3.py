import sys
sys.path.insert(0, "source/utilities")
from task import *
from itertools import product

def make_tasks(verbose=False):
    """
    Creates tasks for the following experiments:
    - SGU
    - SGU + decay
    - SGU + momentum
    - SGU + momentum + decay

    - AdaDelta
    - AdaDelta + momentum

    - RMSProp
    - RMSProp + decay
    - RMSProp + decay + momentum

    - Adam
    """

    batch_size = 100
    model_dir = "models/svhn_3x3_batch_100"

    """
    Some notes:
    - 0.1 is close to the highest learning rate that worked for RMSProp.
    - The values used for beta_1 and beta_1 are taken from the ADAM paper.
    """
    rmsprop_lr_list = [0.001, 0.0008, 0.0006, 0.0004]
    decay_list      = [0.9, 0.95, 0.999]
    mom_list        = [0.9, 0.95, 0.999]
    epsilon_list    = [1e-9, 1e-10, 1e-11]
    lambda_list     = [1 - 1e-9, 1 - 1e-10, 1 - 1e-11]
    beta_1_list     = [0, 0.8, 0.9]
    beta_2_list     = [0.99, 0.999]

    default_rmsprop_lr = 0.001
    default_epsilon    = 1e-10
    default_lambda     = 1 - 1e-10

    tasks = []

    def print_config(name, desc):
        if verbose:
            print("{}\t\"{}\"".format(name, desc))

    for index, epsilon in enumerate(epsilon_list):
        name = "adadelta_eps_" + str(index + 1)
        print_config(name, "eps = {}".format(epsilon))

        tasks.append(Task(name=name, model_dir=model_dir,
            batch_size=str(batch_size), opt_method="adadelta", lr=str(1),
            epsilon=str(epsilon)))

    for index, decay in enumerate(decay_list):
        name = "adadelta_decay_" + str(index + 1)
        print_config(name, "decay = {}".format(decay))

        tasks.append(Task(name=name, model_dir=model_dir,
            batch_size=str(batch_size), opt_method="adadelta",
            lr=str(1), decay=str(decay), epsilon=str(default_epsilon)))

    for index, params in enumerate(product(decay_list, mom_list)):
        decay, mom = params
        name = "adadelta_decay_mom_" + str(index + 1)
        print_config(name, "decay = {}, mom = {}".format(decay, mom))

        tasks.append(Task(name=name, model_dir=model_dir,
            batch_size=str(batch_size), opt_method="adadelta", lr=str(1),
            decay=str(decay), mom=str(mom), mom_type="nag",
            epsilon=str(default_epsilon)))

    for index, epsilon in enumerate(epsilon_list):
        name = "rmsprop_eps_" + str(index + 1)
        print_config(name, "eps = {}".format(epsilon))

        tasks.append(Task(name=name, model_dir=model_dir,
            batch_size=str(batch_size), opt_method="rmsprop",
            lr=str(default_rmsprop_lr), epsilon=str(epsilon)))

    for index, params in enumerate(product(rmsprop_lr_list, decay_list)):
        lr, decay = params
        name = "rmsprop_decay_" + str(index + 1)
        print_config(name, "lr = {}, decay = {}".format(lr, decay))

        tasks.append(Task(name=name, model_dir=model_dir,
            batch_size=str(batch_size), opt_method="rmsprop",
            lr=str(lr), decay=str(decay), epsilon=str(default_epsilon)))

    for index, params in enumerate(product(rmsprop_lr_list, decay_list, mom_list)):
        lr, decay, mom = params
        name = "rmsprop_decay_mom_" + str(index + 1)
        print_config(name, "lr = {}, decay = {}, mom = {}".format(lr, decay, mom))

        tasks.append(Task(name=name, model_dir=model_dir,
            batch_size=str(batch_size), opt_method="rmsprop",
            lr=str(lr), decay=str(decay), mom=str(mom), mom_type="nag",
            epsilon=str(default_epsilon)))

    for index, params in enumerate(product(epsilon_list, lambda_list)):
        epsilon, lambda_ = params
        name = "adam_eps_lambda_" + str(index + 1)
        print_config(name, "eps = {}, lambda = {}".format(epsilon, lambda_))

        tasks.append(Task(name=name, model_dir=model_dir,
            batch_size=str(batch_size), opt_method="adam",
            lr=str(default_rmsprop_lr), epsilon=str(epsilon),
            lambda_=str(lambda_)))

    for index, params in enumerate(product(rmsprop_lr_list, beta_1_list, beta_2_list)):
        lr, beta_1, beta_2 = params
        name = "adam_lr_beta1_beta2_" + str(index + 1)
        print_config(name, "lr = {}, beta_1 = {}, beta_2 = {}".format( \
            lr, beta_1, beta_2))

        tasks.append(Task(name=name, model_dir=model_dir,
            batch_size=str(batch_size), opt_method="adam",
            lr=str(lr), epsilon=str(default_epsilon),
            lambda_=str(default_lambda), beta_1=str(beta_1),
            beta_2=str(beta_2)))

    return tasks

def initialize_workers(local_gpu_count, remote_gpu_count):
    workers = []
    local_command = ["th", "source/drivers/svhn_3x3.lua"]
    remote_command = ["bash", "source/utilities/th_julie.sh", "source/drivers/svhn_3x3.lua"]

    for i in range(1, local_gpu_count + 1):
        workers.append(Worker(local_command + ["-device", str(i)] + \
            ["-max_epochs", "80"]))

    # For some reason, remote commands can get killed before they finish. Maybe
    # this is because of the connection being seen as idle? I don't have time to
    # debug this now.
    #
    #for i in range(1, remote_gpu_count + 1):
    #    workers.append(Worker(remote_command + ["-device", str(i)]))

    return TaskManager(workers)

def run_tasks(tasks):
    gpu_count = 2
    manager = initialize_workers(gpu_count, gpu_count)

    num_epochs    = 80
    sec_per_epoch = 10 + 20 / 5
    time_est = (len(tasks) * num_epochs * sec_per_epoch) / 3600 / gpu_count
    print("Number of tasks: {}.".format(len(tasks)))
    print("Estimated time to completion: {} hours.".format(time_est))

    for index, task in enumerate(tasks):
        print("Submitting task {}/{} ({}).".format(index + 1, len(tasks), task.name))
        manager.submit(task)

    manager.wait()
    print("Done.")

def start():
    tasks = make_tasks()
    run_tasks(tasks)

def finish_remaining_tasks():
    tasks = make_tasks()
    lines = open("logs/completed_tasks.log").readlines()
    completed_tasks = []
    for task in lines:
        completed_tasks.append(task.strip())

    new_tasks =  []
    for task in tasks:
        if not task.name in completed_tasks:
            new_tasks.append(task)
        else:
            print("Skipping task {}.".format(task))
    run_tasks(new_tasks)


make_tasks(True)
#start()
#finish_remaining_tasks()
