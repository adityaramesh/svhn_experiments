import sys
sys.path.insert(0, "source/utilities")
from task import *
from itertools import product

def make_tasks():
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

    """
    Some notes:
    - 0.4 is close to the highest learning rate that worked for SGU.
    - 0.1 is close to the highest learning rate that worked for RMSProp.
    - The values used for beta_1 and beta_1 are taken from the ADAM paper.
    """
    sgu_lr_list     = [0.1, 0.2, 0.3, 0.4]
    rmsprop_lr_list = [0.001, 0.0008, 0.0006, 0.0004]
    lr_decay_list   = [1e-4, 1e-6, 1e-8]
    decay_list      = [0.9, 0.95, 0.999]
    mom_list        = [0.9, 0.95, 0.999]
    epsilon_list    = [1e-8, 1e-10, 1e-12]
    lambda_list     = [1 - 1e-8, 1 - 1e-10, 1 - 1e-12]
    beta_1_list     = [0, 0.8, 0.9, 0.95]
    beta_2_list     = [0.8, 0.9, 0.99, 0.999]

    default_rmsprop_lr = 0.001
    default_epsilon    = 1e-10
    default_lambda     = 1 - 1e-10

    tasks = []

    for index, lr in enumerate(sgu_lr_list):
        tasks.append(Task(name="sgu_" + str(index + 1),
            batch_size=str(batch_size), opt_method="sgu", lr=str(lr)))

    for index, params in enumerate(product(sgu_lr_list, lr_decay_list)):
        lr, decay = params
        tasks.append(Task(name="sgu_lr_decay_" + str(index + 1),
            batch_size=str(batch_size), opt_method="sgu", lr=str(lr),
            lr_sched="gentle_decay", lr_decay=str(decay)))

    for index, params in enumerate(product(sgu_lr_list, mom_list)):
        lr, mom = params
        tasks.append(Task(name="sgu_lr_mom_" + str(index + 1),
            batch_size=str(batch_size), opt_method="sgu", lr=str(lr),
            mom=str(mom), mom_type="nag"))

    for index, params in enumerate(product(sgu_lr_list, mom_list, lr_decay_list)):
        lr, mom, decay = params
        tasks.append(Task(name="sgu_lr_mom_decay_" + str(index + 1),
            batch_size=str(batch_size), opt_method="sgu", lr=str(lr),
            mom=str(mom), mom_type="nag", lr_sched="gentle_decay",
            lr_decay=str(decay)))

    for index, epsilon in enumerate(epsilon_list):
        tasks.append(Task(name="adadelta_eps_" + str(index + 1),
            batch_size=str(batch_size), opt_method="adadelta", lr=str(1),
            epsilon=str(epsilon)))

    for index, decay in enumerate(decay_list):
        tasks.append(Task(name="adadelta_decay_" + str(index + 1),
            batch_size=str(batch_size), opt_method="adadelta", lr=str(1),
            decay=str(decay), epsilon=str(default_epsilon)))

    for index, params in enumerate(product(decay_list, mom_list)):
        decay, mom = params
        tasks.append(Task(name="adadelta_decay_mom_" + str(index + 1),
            batch_size=str(batch_size), opt_method="adadelta", lr=str(1),
            decay=str(decay), mom=str(mom), mom_type="nag",
            epsilon=str(default_epsilon)))

    for index, epsilon in enumerate(epsilon_list):
        tasks.append(Task(name="rmsprop_eps_" + str(index + 1),
            batch_size=str(batch_size), opt_method="rmsprop",
            lr=str(default_rmsprop_lr), epsilon=str(epsilon)))

    for index, params in enumerate(product(rmsprop_lr_list, decay_list)):
        lr, decay = params
        tasks.append(Task(name="rmsprop_decay_" + str(index + 1),
            batch_size=str(batch_size), opt_method="rmsprop", lr=str(lr),
            decay=str(decay), epsilon=str(default_epsilon)))

    for index, params in enumerate(product(rmsprop_lr_list, decay_list, mom_list)):
        lr, decay, mom = params
        tasks.append(Task(name="rmsprop_decay_mom_" + str(index + 1),
            batch_size=str(batch_size), opt_method="rmsprop", lr=str(lr),
            decay=str(decay), mom=str(mom), mom_type="nag",
            epsilon=str(default_epsilon)))

    for index, params in enumerate(product(epsilon_list, lambda_list)):
        epsilon, lambda_ = params
        tasks.append(Task(name="adam_eps_lambda_" + str(index + 1),
            batch_size=str(batch_size), opt_method="adam",
            lr=str(default_rmsprop_lr), epsilon=str(epsilon),
            lambda_=str(lambda_)))

    for index, params in enumerate(product(rmsprop_lr_list, beta_1_list, beta_2_list)):
        lr, beta_1, beta_2 = params
        tasks.append(Task(name="adam_lr_beta1_beta2_" + str(index + 1),
            batch_size=str(batch_size), opt_method="adam", lr=str(lr),
            epsilon=str(default_epsilon), lambda_=str(default_lambda),
            beta_1=str(beta_1), beta_2=str(beta_2)))

    return tasks

def initialize_workers(local_gpu_count, remote_gpu_count):
    workers = []
    local_command = ["th", "source/drivers/svhn_5x5.lua"]
    remote_command = ["bash", "source/utilities/th_julie.sh", "source/drivers/svhn_5x5.lua"]

    for i in range(1, local_gpu_count + 1):
        workers.append(Worker(local_command + ["-device", str(i)]))
    for i in range(1, remote_gpu_count + 1):
        workers.append(Worker(remote_command + ["-device", str(i)]))
    return TaskManager(workers)

def run_tasks():
    gpu_count = 2
    tasks = make_tasks()
    manager = initialize_workers(gpu_count, gpu_count)

    num_epochs    = 50
    sec_per_epoch = 10 + 10 / 3
    time_est = (len(tasks) * num_epochs * sec_per_epoch) / 3600 / gpu_count
    print("Number of tasks: {}.".format(len(tasks)))
    print("Estimated time to completion: {} hours.".format(time_est))

    for index, task in enumerate(tasks):
        print("Submitting task {}/{} ({}).".format(index + 1, len(tasks), task.name))
        manager.submit(task)

    manager.wait()
    print("Done.")

run_tasks()
