import time
import sys
from subprocess import Popen, DEVNULL

class Worker:
    def __init__(self, base_command):
        self.base_command = base_command
        self.command = None
        self.process = None
        self.return_code = None

    def busy(self):
        if not self.process:
            return False

        self.return_code = self.process.poll()
        if self.return_code is None:
            return True

        if self.return_code != 0:
            print("Warning: command {} returned exit code {}.". \
                format(self.command, self.return_code), file=sys.stderr)
        return False

    def submit(self, arguments):
        if self.busy():
            raise RuntimeError("Attempt to submit task while busy.")

        self.command = self.base_command + arguments
        self.process = Popen(self.command, stdout=DEVNULL, stderr=DEVNULL)
        self.return_code = None

class TaskManager:
    def __init__(self, workers):
        self.workers = workers
        self.cur_pos = 0

    def submit(self, task):
        pos = self.cur_pos
        while self.workers[pos].busy():
            pos = (pos + 1) % len(self.workers)
            if pos == self.cur_pos:
                time.sleep(1)

        self.workers[pos].submit(task.arguments())
        self.cur_pos = (pos + 1) % len(self.workers)

    def wait(self):
        for worker in self.workers:
            while worker.busy():
                time.sleep(1)

class Task:
    def __init__(self, name, batch_size, opt_method, lr=None, lr_sched=None, \
        lr_decay=None, mom_type=None, mom=None, decay=None, epsilon=None,    \
        lambda_=None, beta_1=None, beta_2=None):

        self.name       = name
        self.batch_size = batch_size
        self.model_dir  = "models/batch_" + str(self.batch_size)
        self.opt_method = opt_method
        self.lr         = lr
        self.lr_sched   = lr_sched
        self.lr_decay   = lr_decay
        self.mom_type   = mom_type
        self.mom        = mom
        self.decay      = decay
        self.epsilon    = epsilon
        self.lambda_    = lambda_
        self.beta_1     = beta_1
        self.beta_2     = beta_2

        opt_methods = ["sgu", "adadelta", "rmsprop", "adam"]
        if not opt_method in opt_methods:
            raise RuntimeError("Invalid optimization method \"{}\".".format(opt_method))
        if opt_method != "rmsprop" and lr == None:
            raise RuntimeError("Learning rate required for optimization " \
                "method \"{}\".".format(opt_method))

    def arguments(self):
        args = ["-task", "replace", "-model", self.name, "-model_dir", self.model_dir, \
            "-batch_size", str(self.batch_size), "-opt_method", self.opt_method]

        if self.lr is not None:
            args.extend(["-learning_rate", str(self.lr)])
        if self.lr_sched is not None:
            args.extend(["-learning_rate_schedule", str(self.lr_sched)])
        if self.lr_decay is not None:
            args.extend(["-learning_rate_decay", str(self.lr_decay)])
        if self.mom_type is not None:
            args.extend(["-momentum_type", str(self.mom_type)])
        if self.mom is not None:
            args.extend(["-momentum", str(self.mom)])
        if self.decay is not None:
            args.extend(["-decay", str(self.decay)])
        if self.epsilon is not None:
            args.extend(["-epsilon", str(self.epsilon)])
        if self.lambda_ is not None:
            args.extend(["-lambda", str(self.lambda_)])
        if self.beta_1 is not None:
            args.extend(["-beta_1", str(self.beta_1)])
        if self.beta_2 is not None:
            args.extend(["-beta_2", str(self.beta_2)])
        return args
