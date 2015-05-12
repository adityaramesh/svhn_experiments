<!--
  ** File Name: README.md
  ** Author:    Aditya Ramesh
  ** Date:      05/08/2015
  ** Contact:   _@adityaramesh.com
-->

# Overview

Experiments involving CNNs on the SVHN data set.

## Goals

Our experiments will involve two architectures:

  1. The [5x5 filter architecture](source/modeles/cnn_5x5.lua).
  2. The [3x3 filter architecture](source/modeles/cnn_3x3.lua).

The 5x5 architecture converges relatively quickly, so our results for this model
should include the validation error *at convergence*. The 3x3 architecture takes
a long time to converge. It is not practical for us to measure the validation
error at converge for each choice of hyperparameters. Instead, I propose that we
measure the best training and test error achieved by the optimization algorithm
after *30 training epochs*.

## Getting Started

Before attempting to train any models, you must first download the preprocessed
data files. I should have sent you an email with the URLs of these files, along
with instructions as to where to put them.

To train a model, use the command

	th <driver_file> -task <task> -model <model_name> -device <n>

where `<driver_file>` is the name of the driver file, `<task>` is either
`resume` or `replace`, `<model_name>` is the name you wish to give to the model,
and `<n>` is the ordinal of the GPU that you wish to use. After the script
executes, a directory with the name `<model_name>` will be created in `models`.
`resume` is used to resume training from the last epoch if the script has been
terminated (e.g. by Control-C). `replace` *removes* the directory
`models/<model_name>` and creates everything from scratch.

For example, to train the 5x5 model with the baseline configuration, type

	th source/drivers/svhn_5x5_baseline.lua -task replace -model 5x5_baseline -device 1

All of the model pertaining to your model will be stored in
`models/<model_name>`. This includes the following:

  - The current model and optimization state.
  - The model and optimization state that achieved the best training error.
  - The model and optimization state that achieved the best validation error.
  - A CSV file containing the training scores.
  - A CSV file containing the validation scores.

## How Driver Scripts Work

The driver scripts in `source/drivers` each include a file that defines
`get_model_info`, and another file that defines `get_train_info`. The former
function provides the definition of the model; the latter provides the
definition of the optimization algorithm. Both of these are supplied to `run`, a
function in `source/utilities/run_model.lua` that initiates training.

## Available Optimization Algorithms

Torch's [`optim` package](https://github.com/torch/optim) implements all of the
algorithms available in [`sopt`](https://github.com/adityaramesh/torch_utils).
However, `sopt` adds NAG and the ability to use arbitrary decay schedules for
many parameters.

`sopt` implements the following algorithms:

  - SGU (stochastic gradient update) (+ CM, NAG): see `source/train/sgu_100.lua`.
  - RMSProp (+ NAG): see `source/train/rmsprop_100.lua`.
  - AdaDelta (+ NAG): see `source/train/adadelta_100.lua`.

I propose that we also use Adam, which is available from Torch's `optim`
package: see `source/train/adam_100.lua`.

## What We Should Change

- For SGU:
  - The learning rate. Change the learning rate, and also try using
  `sopt.gentle_decay` instead of `sopt.constant`.
  - The momentum parameter.
  - Effect of not using momentum (it's currently used by default; to disable
  momentum, use `momentum_type = sopt.none`).
  - I *do not* recommend using `momentum_type = sopt.cm`; CM has always
  performed worse than NAG in my experience, and it uses twice as many function
  evaluations.
  - The batch size: 50 and 200 are also worth trying.

- For RMSProp:
  - Everything mentioned for SGU.
  - Epsilon.
  - The decay (this is different from the momentum).

- For AdaDelta:
  - Everything mentioned for RMSProp (AdaDelta also has epsilon and decay
  parameters).

- For Adam:
  - We have to use Torch's implementation, because I didn't implement Adam. This
  means that we can't use NAG or decay the learning rate. But everything else
  mentioned for RMSProp applies.
  - The `beta1` parameter.
  - The `beta2` parameter.
  - The `epsilon` parameter.
  - The `lambda` parameter.
