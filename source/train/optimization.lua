require "torch"
require "optim"

package.path = package.path .. ";./torch_utils/?.lua"
require "torch_utils/sopt"

function get_train_info(opt)
	local opt_method = {}
	if opt.opt_method == "sgu" then
		opt_method = SGUOptimizer
	--elseif opt.opt_method == "sgu_eig" then
	--	opt_method = sopt.sgu_eig
	--elseif opt.opt_method == "sgu_eig_info" then
	--	opt_method = sopt.sgu_eig_info
	--elseif opt.opt_method == "adadelta" then
	--	opt_method = sopt.adadelta
	--elseif opt.opt_method == "rmsprop" then
	--	opt_method = sopt.rmsprop
	--elseif opt.opt_method == "adam" then
	--	opt_method = optim.adam
	else
		print("Invalid optimization method \"" .. opt.opt_method ..  "\".")
	end

	local batch_size = opt.batch_size
	local learning_rate = opt.learning_rate_schedule == "constant" and
		sopt.constant(opt.learning_rate) or
		sopt.gentle_decay(opt.learning_rate, opt.learning_rate_decay)
	local momentum_type = opt.momentum_type == "none" and
		sopt.none or sopt.nag
	local momentum = opt.momentum
	local decay = opt.decay
	local epsilon = opt.epsilon
	local lambda = opt.lambda
	local beta_1 = opt.beta_1
	local beta_2 = opt.beta_2

	return {
		opt_state = {
			learning_rate = learning_rate,
			learningRate = opt.learning_rate,
			momentum_type = momentum_type,
			momentum = sopt.constant(momentum),
			decay = sopt.constant(decay),
			epsilon = epsilon,
			lambda = lambda,
			beta1 = beta_1,
			beta2 = beta_2
		},
		opt_method = opt_method,
		batch_size = batch_size
	}
end
