require "torch"

package.path = package.path .. ";./torch_utils/?.lua"
require "torch_utils/sopt"

function get_train_info()
	return {
		opt_state = {
			learning_rate = sopt.constant(0.1),
			momentum = sopt.constant(0.95),
			momentum_type = sopt.nag
		},
		opt_method = sopt.sgu,
		batch_size = 100
	}
end
