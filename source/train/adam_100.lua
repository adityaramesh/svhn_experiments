require "torch"
require "optim"

function get_train_info()
	return {
		opt_state = {
			learningRate = 0.001,
			beta1 = 0.9,
			beta2 = 0.999,
			epsilon = 1e-8,
			lambda = 1 - 1e-8
		},
		opt_method = optim.adam,
		batch_size = 100
	}
end
