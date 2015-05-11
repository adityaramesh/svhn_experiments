require "torch"

package.path = package.path .. ";./torch_utils/?.lua"
require "torch_utils/sopt"
require "torch_utils/model_utils"

function get_train_info()
	print("Using ada-delta with rho:" .. model_utils.adadelta_rho .. " eps:" .. model_utils.adadelta_eps)
	print("rho type: " .. type(model_utils.adadelta_rho) .. " eps type: " .. type(model_utils.adadelta_eps))
	return {
		opt_state = {
			epsilon = model_utils.adadelta_eps,
			decay = sopt.constant(model_utils.adadelta_rho),
			learning_rate = sopt.constant(1),
			momentum = sopt.constant(0.95),
			momentum_type = sopt.none
		},
		opt_method = sopt.adadelta,
		batch_size = 100
	}
end
