<!--
  ** File Name: README.md
  ** Author:    Aditya Ramesh
  ** Date:      05/08/2015
  ** Contact:   _@adityaramesh.com
-->

# Overview

Experiments involving CNNs on the SVHN data set.

## TODO

- Use the power method to compute both the positive and negative eigenvalues of
largest magnitude. This would also tell us if the Hessian is positive definite.
- Log extra information involving the eigenvalue within the optimizer.

## Useful Crap

    th source/drivers/svhn_5x5.lua -device 1 -task replace -model adadelta
	-model_dir best_5x5_batch_100 -batch_size 100 -opt_method adadelta
	-learning_rate 1 -decay 0.999 -max_epochs 100 2>&1 | tee sgu_output.log


