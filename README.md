<!--
  ** File Name: README.md
  ** Author:    Aditya Ramesh
  ** Date:      05/08/2015
  ** Contact:   _@adityaramesh.com
-->

# Overview

Experiments involving CNNs on the SVHN data set.

## TODO

- Check whether using the top-k results provides any benefit.
	- If not, then allow failures to compute eigenvalues.
	- If it does, see if we can change `3 * inner_iters` to `inner_iters`
	without any performance reduction. Then refactor the code.
