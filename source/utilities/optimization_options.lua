require "torch_utils/model_utils"

function optimization_options(cmd)
	model_utils.default_options(cmd)
	cmd:option("-max_epochs", 50)
	cmd:option("-opt_method", "sgu", "sgu | adadelta | rmsprop | adam")
	cmd:option("-batch_size", 100)
	cmd:option("-learning_rate", 1)
	cmd:option("-learning_rate_schedule", "constant", "constant | gentle_decay")
	cmd:option("-learning_rate_decay", 1e-7)
	cmd:option("-momentum_type", "none", "none | nag")
	cmd:option("-momentum", 0.95)
	cmd:option("-decay", 0.95)
	cmd:option("-epsilon", 1e-8)
	cmd:option("-lambda", 1 - 1e-8)
	cmd:option("-beta_1", 0.9)
	cmd:option("-beta_2", 0.999)
end
