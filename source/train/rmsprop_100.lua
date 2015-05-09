require "torch"

package.path = package.path .. ";./torch_utils/?.lua"
require "torch_utils/sopt"

function get_train_info()
	return {
		opt_state = {
			learning_rate = sopt.constant(0.001),
			momentum = sopt.constant(0.95),
			momentum_type = sopt.nag
		},
		opt_method = sopt.rmsprop,
		batch_size = 100
	}
end
