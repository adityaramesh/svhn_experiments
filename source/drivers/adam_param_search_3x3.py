import sys
sys.path.insert(0, "source/utilities")
from task import *
from itertools import product

def make_tasks(verbose=False):
    batch_size = 100
    model_dir = "models/svhn_adam_3x3_batch_100"

    lr_list = [0.0004, 0.0002, 0.00007, 0.00004, 0.00001]
    beta_1_list = [0.8, 0.6, 0.4, 0.2]
    beta_2 = 0.9999
    epsilon = 1e-11
    lambda_ = 1 - 1e-11
    tasks = []

    def print_config(name, desc):
        if verbose:
            print("{}\t\"{}\"".format(name, desc))

    for index, params in enumerate(product(lr_list, beta_1_list)):
        lr, beta_1 = params
        name = "adam_" + str(index + 1)
        print_config(name, "lr = {}, beta_1 = {}".format(lr, beta_1))

        tasks.append(Task(name=name, model_dir=model_dir,
            batch_size=str(batch_size), opt_method="adam", lr=str(lr),
            epsilon=str(epsilon), lambda_=str(lambda_), beta_1=str(beta_1),
            beta_2=str(beta_2)))

    return tasks

def initialize_workers(local_gpu_count, remote_gpu_count):
    workers = []
    local_command = ["th", "source/drivers/svhn_3x3.lua"]
    remote_command = ["bash", "source/utilities/th_julie.sh", "source/drivers/svhn_3x3.lua"]

    for i in range(1, local_gpu_count + 1):
        workers.append(Worker(local_command + ["-device", str(i)] + \
            ["-max_epochs", "100"]))
    return TaskManager(workers)

def run_tasks(tasks):
    gpu_count = 2
    manager = initialize_workers(gpu_count, gpu_count)

    num_epochs    = 100
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


start()
#finish_remaining_tasks()
