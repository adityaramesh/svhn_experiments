require "torch"
require "optim"
require "xlua"

require "torch_utils/model_utils"
require "source/train/optimization"
require "source/utilities/optimization_options"
require "source/utilities/csv_logger"
require "source/utilities/eigenvalue_estimator"

local function sample_max_eigenvalue(data, context, paths, info)
	local train_size  = data.inputs:size(1)
	local batch_size  = info.train.batch_size
	local params      = context.params
	local perm        = torch.randperm(train_size)

	local samples  = 100
	local min_eigs = {}
	local max_eigs = {}

	for i = 1, 1 + (samples - 1) * batch_size, batch_size do
		print("Iteration " .. (i - 1) / batch_size + 1)

		-- Create the mini-batch.
		local cur_batch_size = math.min(batch_size, train_size - i + 1)
		local inputs = {}
		local targets = {}
		for j = i, i + cur_batch_size - 1 do
			local input = data.inputs[{{perm[j]}}]
			local target = data.targets[{{perm[j]}}]
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		inputs = nn.JoinTable(1):forward(inputs):typeAs(params)
		targets = nn.JoinTable(1):forward(targets):typeAs(params)

		local eig_1, _, eig_2, _ = context.eig_estimator:
			get_min_max_eig(inputs, targets)
		print(eig_1, eig_2)

		if eig ~= nil then
			min_eigs[#min_eigs + 1] = eig_1
			max_eigs[#max_eigs + 1] = eig_2
		end
	end

	--context.logger:log_array("min_eigs", min_eigs)
	--context.logger:log_array("max_eigs", max_eigs)
end

local function do_train_epoch(data, context, paths, info)
	local model       = info.model.model
	local criterion   = info.model.criterion
	local params      = context.params
	local grad_params = context.grad_params
	local confusion   = context.confusion

	local train_size  = data.inputs:size(1)
	local opt_method  = info.train.opt_method
	local opt_state   = info.train.opt_state
	local batch_size  = info.train.batch_size

	if info.train.epoch == nil then
		info.train.epoch = 1
	end

	local perm = torch.randperm(train_size)
	model:training()
	print("Starting training epoch " .. info.train.epoch .. ".")

	for i = 1, train_size, batch_size do
		xlua.progress(i, train_size)

		-- Create the mini-batch.
		local cur_batch_size = math.min(batch_size, train_size - i + 1)
		local inputs = {}
		local targets = {}
		for j = i, i + cur_batch_size - 1 do
			local input = data.inputs[{{perm[j]}}]
			local target = data.targets[{{perm[j]}}]
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		input = nn.JoinTable(1):forward(inputs):typeAs(params)
		target = nn.JoinTable(1):forward(targets):typeAs(params)
		context.optimizer:update(input, target)

		local k = (i - 1) / batch_size + 1
		local iters_per_epoch = train_size / batch_size
		local iters_per_eig_est = iters_per_epoch / 3

		context.logger:log_value("train_epoch", info.train.epoch)
		context.logger:log_value("train_iter", k)

		-- Let `m := iters_per_eig_est`. We want to check whether
		-- `k - 1 = round(q * m)`, where `q` is an integer. After
		-- some manipulation, one gets
		--     floor((k - 1) / ceil(m)) <= q <= ceil((k - 1) / floor(m)).
		-- This allows us to determine whether `k - 1` satisfies the
		-- desired condition after checking only a few integer values.

		--local qmin = math.floor((k - 1) / math.ceil(iters_per_eig_est))
		--local qmax = math.ceil((k - 1) / math.floor(iters_per_eig_est))
		--for q = qmin, qmax do
		--	if k - 1 == math.floor(q * iters_per_eig_est + 0.5) then
		--		sample_max_eigenvalue(data, context, paths, info)
		--		break
		--	end
		--end

		context.logger:flush()
	end

	xlua.progress(train_size, train_size)
	confusion:updateValids()
	local acc = confusion.totalValid
	print("Mean class accuracy (training): " .. 100 * acc .. "%.")
	confusion:zero()

	model_utils.save_train_progress(function(x, y) return x > y end,
		info.train.epoch, acc, paths, info)
	info.train.epoch = info.train.epoch + 1
end

local function do_valid_epoch(data, context, paths, info)
	local model      = info.model.model
	local criterion  = info.model.criterion
	local valid_size = data.inputs:size(1)
	local batch_size = info.train.batch_size
	local params     = context.params
	local confusion  = context.confusion
	model:evaluate()

	print("Performing validation epoch.")
	for i = 1, valid_size, batch_size do
		xlua.progress(i, valid_size)

		-- Create the mini-batch.
		local cur_batch_size = math.min(batch_size, valid_size - i + 1)
		local inputs = {}
		local targets = {}
		for j = i, i + cur_batch_size - 1 do
			local input = data.inputs[{{j}}]
			local target = data.targets[{{j}}]
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		inputs = nn.JoinTable(1):forward(inputs):typeAs(params)
		targets = nn.JoinTable(1):forward(targets):typeAs(params)
		local outputs = model:forward(inputs)
		confusion:batchAdd(outputs, targets)
	end

	xlua.progress(valid_size, valid_size)
	confusion:updateValids()
	local acc = confusion.totalValid
	print("Mean class accuracy (validation): " .. 100 * acc .. "%.")
	confusion:zero()

	if info.train.epoch ~= nil then
		model_utils.save_test_progress(function(x, y) return x > y end,
			info.train.epoch - 1, acc, paths, info)
	end
end

function make_context(info)
	local context = {}
	local log_dir = "logs"
	local log_path = paths.concat(log_dir, info.options.model .. "_output.log")
	context.logger = CSVLogger.create(log_path, {"train_epoch",
		"train_iter"}) --, "min_eigs", "max_eigs"})
		
	local model = info.model.model
	local criterion = info.model.criterion
	context.params, context.grad_params = model:getParameters()
	context.confusion = optim.ConfusionMatrix(10)

	context.grad_func = function(input, target, update_confusion)
		context.grad_params:zero()
		local output = model:forward(input)
		local loss = criterion:forward(output, target)

		if update_confusion then
			context.confusion:batchAdd(output, target)
		end

		model:backward(input, criterion:backward(output, target))
		return loss
	end

	context.eig_estimator = EigenvalueEstimator.create(
		model, context.params, context.grad_params, context.grad_func)
	info.train.opt_state.eig_estimator = context.eig_estimator

	context.optimizer = info.train.opt_method.create(
		model, context.params, context.grad_params, context.grad_func,
		info.train.opt_state, context.logger)

	context.logger:write_header()
	return context
end

function run(model_info_func)
	print("Loading data.")
	local data_dir = "data/preprocessed/"
	local train_data = torch.load(data_dir .. "train_small.t7")
	local valid_data = torch.load(data_dir .. "test.t7")
	local do_train, _, paths, info = model_utils.restore(
		model_info_func, get_train_info, optimization_options)

	print("Configuration options:")
	print(info.options)

	-- Since the training epoch count is only incremented after
	-- serialization, the actual training epoch will always be one greater
	-- than the number that has been serialized.
	if info.train.epoch ~= nil then
		info.train.epoch = info.train.epoch + 1
	end

	local context = make_context(info)
	local max_epochs = info.options.max_epochs or 1000
	print("")

	if do_train then
		while info.train.epoch == nil or info.train.epoch <= max_epochs do
			do_train_epoch(train_data, context, paths, info)
			print("")

			local cur_epoch = info.train.epoch - 1
			if cur_epoch % 3 == 0 or cur_epoch >= max_epochs - 5 then
				do_valid_epoch(valid_data, context, paths, info)
				print("")
			end
		end
	else
		do_valid_epoch(valid_data, context, paths, info)
	end
end
