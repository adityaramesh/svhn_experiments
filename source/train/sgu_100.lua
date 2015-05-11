require "torch"

package.path = package.path .. ";./torch_utils/?.lua"
require "torch_utils/sopt"
require "torch_utils/model_utils"

function get_train_info()
	return {
		opt_state = {
			learning_rate = sopt.constant(model_utils.sgd_eps),
			momentum = sopt.constant(0.95),
			momentum_type = sopt.none
		},
		opt_method = sopt.sgu,
		batch_size = 100
	}
end
