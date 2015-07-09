require "torch"
require "optim"
require "xlua"

require "torch_utils/model_utils"
require "source/train/optimization"
require "source/utilities/optimization_options"

local function local_grid_search(eig_func, iters, gran, tol, alpha)
	assert(gran > 0 and gran < 1)
	assert(alpha > 0 and alpha < 1)

	local a = math.max(alpha - 10 * gran, gran)
	local b = alpha + 10 * gran
	local best_alpha = 0
	local best_skew = 1
	local eig = 0

	for cur_alpha = a, b + gran, gran do
		local cur_skew, cur_eig = eig_func(cur_alpha, iters)

		if cur_skew < best_skew then
			if cur_skew < tol then
				return cur_alpha, cur_skew, cur_eig
			end

			best_alpha = cur_alpha
			best_skew = cur_skew
			eig = cur_eig
		end
	end

	return best_alpha, best_skew, eig
end

local function iterative_grid_search(eig_func, outer_iters, inner_iters, tol, alpha, skew)
	assert(tol > 0 and tol < 1)
	assert(alpha > 0 and alpha < 1)

	local exp = math.floor(math.log(alpha) / math.log(10))
	local gran = math.pow(10, exp - 1)
	local best_alpha = alpha
	local best_skew = skew
	local eig = 0

	for i = 1, outer_iters do
		local cur_alpha, cur_skew, cur_eig =
			local_grid_search(eig_func, inner_iters, gran, tol, best_alpha)

		if cur_skew < best_skew then
			if cur_skew < tol then
				return cur_alpha, cur_skew, cur_eig
			end

			best_alpha = cur_alpha
			best_skew = cur_skew
			eig = cur_eig
		end
		gran = gran / 10
	end

	return best_alpha, best_skew, eig
end

--
-- TODO: Refactor this later.
--
local function estimate_max_eigenvalue(data, context, paths, info)
	local model       = info.model.model
	local criterion   = info.model.criterion
	local params      = context.params
	local grad_params = context.grad_params

	local train_size  = data.inputs:size(1)
	local batch_size  = info.train.batch_size
	local perm        = torch.randperm(train_size)

	local outer_iters = 100
	local inner_iters = 5
	assert(outer_iters <= batch_size)

	-- This value of `tol` is actually quite forgiving. If `tol` is much
	-- higher than 1e-8, then the corresponding eigenvalue will only be
	-- correct to within an order of magnitude. I wouldn't recommend making
	-- this any smaller.
	local tol = 1e-6

	local alpha_list  = {
		-- For some reason, 6.56e-8 seems to give good results for a
		-- large fraction of the finite differences.
		6.56e-8,
		1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8, 9e-8,
		1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7,
		1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9, 9e-9,
		1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6,
		1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5
	}

	local phi        = torch.CudaTensor(params:size(1))
	local init_phi   = torch.randn(params:size(1)):cuda()
	local tmp_grad   = torch.CudaTensor(params:size(1))
	local tmp_params = params:clone()
	local eigs       = {}

	for i = 1, 1 + (outer_iters - 1) * batch_size, batch_size do
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

		inputs = nn.JoinTable(1):forward(inputs):cuda()
		targets = nn.JoinTable(1):forward(targets):cuda()

		-- Define the function to obtain the model's output and gradient
		-- with respect to parameters.
		local grad_func = function(x)
			if x ~= params then
				params:copy(x)
			end
			grad_params:zero()

			local outputs = model:forward(inputs)
			local loss = criterion:forward(outputs, targets)
			model:backward(inputs, criterion:backward(outputs, targets))
			return loss, grad_params
		end

		local eig_func = function(alpha, iters)
			-- "Hot-starting" the iterations using the previous
			-- value of `phi` actually seems to retard progress.
			-- Instead, we use the saved value of `init_phi`.
			phi:copy(init_phi)

			local proj = 0
			local skew = 0
			local norm = 0

			for j = 1, iters do
				-- Because of finite precision arithmetic, we
				-- can't undo the action of adding `-alpha *
				-- phi` to `params` by subtracting the same
				-- quantity. Instead, we save `params` and
				-- restore it after bprop.
				params:add(-alpha, phi)
				grad_func(params)
				tmp_grad:copy(grad_params)

				params:copy(tmp_params)
				params:add(alpha, phi)
				grad_func(params)
				params:copy(tmp_params)
				grad_params:add(-1, tmp_grad):div(2 * alpha)

				-- How close is `phi` to being an eigenvector?
				norm = grad_params:norm()
				proj = phi:dot(grad_params) / norm
				skew = math.min(math.abs(proj - 1), math.abs(proj + 1))
				if skew < tol then
					break
				end

				phi:copy(grad_params)
				phi:div(norm)
			end

			norm = proj > 0 and norm or -norm
			return skew, norm
		end

		local best_alpha = 0
		local best_skew = 1
		local eig = 0

		local isnan = function(x) return x ~= x end

		-- Note: we might get this to run faster by reinitializing
		-- `init_phi` sooner if we aren't able to compute the maximum
		-- eigenvalue.
		local grid_search_func = function()
			local results = {}

			-- First try all of the `alpha` values that are in
			-- `alpha_list` for the finite difference.
			for j = 1, #alpha_list do
				local cur_alpha = alpha_list[j]
				local cur_skew, cur_eig = eig_func(cur_alpha, inner_iters)

				if not isnan(cur_skew) then
					results[cur_skew] = cur_alpha

					if cur_skew < best_skew then
						best_alpha = cur_alpha
						best_skew = cur_skew
						eig = cur_eig

						if best_skew < tol then
							break
						end
					end
				end
			end

			-- If we weren't able to find a good value for `alpha`,
			-- then run a nested grid search to try to hone in on a
			-- better value.
			if best_skew >= tol then
				local accuracies = {}
				for k in pairs(results) do
					table.insert(accuracies, k)
				end
				table.sort(accuracies)

				for j = 1, math.min(#accuracies, 5) do
					local cur_skew = accuracies[j]
					local cur_alpha = results[cur_skew]

					local new_alpha, new_skew, new_eig =
						iterative_grid_search(eig_func,
							3, inner_iters, tol,
							cur_alpha, cur_skew)

					if new_skew < best_skew then
						best_alpha = new_alpha
						best_skew = new_skew
						eig = new_eig

						if best_skew < tol then
							break
						end
					end
				end
			end
		end

		grid_search_func()

		-- If we still weren't able to compute the maximum eigenvalue
		-- using the nested grid search, then reinitialize `init_phi`
		-- and try again.
		if best_skew >= tol then
			for j = 1, 5 do
				best_alpha = 0
				best_skew = 1
				eig = 0

				init_phi:copy(torch.randn(params:size(1)))
				grid_search_func()

				if best_skew < tol then
					break
				end
			end
		end

		if best_skew < tol then
			print("Found maximum eigenvalue: " .. eig .. ".")
			eigs[#eigs + 1] = eig
		else
			print("Failed to find maximum eigenvalue.")
			print("Best skew: " .. best_skew .. ".")
			print("Best alpha: " .. best_alpha .. ".")
		end
	end

	eigs = torch.Tensor(eigs)
	print(eigs)
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

		inputs = nn.JoinTable(1):forward(inputs):cuda()
		targets = nn.JoinTable(1):forward(targets):cuda()

		-- Define the function to obtain the model's output and gradient
		-- with respect to parameters.
		local grad_func = function(x)
			if x ~= params then
				params:copy(x)
			end
			grad_params:zero()

			local outputs = model:forward(inputs)
			local loss = criterion:forward(outputs, targets)
			model:backward(inputs, criterion:backward(outputs, targets))
			confusion:batchAdd(outputs, targets)
			return loss, grad_params
		end

		opt_method(grad_func, params, opt_state)

		local k = (i - 1) / batch_size + 1
		local iters_per_epoch = train_size / batch_size
		local iters_per_eig_est = iters_per_epoch / 3

		-- Let `m := iters_per_eig_est`. We want to check whether
		-- `k - 1 = round(q * m)`, where `q` is an integer. After
		-- some manipulation, one gets
		--     floor((k - 1) / ceil(m)) <= q <= ceil((k - 1) / floor(m)).
		-- This allows us to determine whether `k - 1` satisfies the
		-- desired condition after checking only a few integer values.

		local qmin = math.floor((k - 1) / math.ceil(iters_per_eig_est))
		local qmax = math.ceil((k - 1) / math.floor(iters_per_eig_est))
		for q = qmin, qmax do
			if k - 1 == math.floor(q * iters_per_eig_est + 0.5) then
				print(k)
				estimate_max_eigenvalue(data, context, paths, info)
				break
			end
		end
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

		inputs = nn.JoinTable(1):forward(inputs):cuda()
		targets = nn.JoinTable(1):forward(targets):cuda()
		local outputs = model:forward(inputs)
		confusion:batchAdd(outputs, targets)

		if i % 33 == 0 then
			-- This helps prevent out of memory errors on the GPU.
			collectgarbage()
		end
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

function run(model_info_func)
	print("Loading data.")
	local data_dir = "data/preprocessed/"
	local train_data = torch.load(data_dir .. "train_small.t7")
	local valid_data = torch.load(data_dir .. "test.t7")

	local do_train, _, paths, info = model_utils.restore(
		model_info_func, get_train_info, optimization_options)

	print("Configuration options:")
	print(info.options)

	if info.train.epoch ~= nil then
		info.train.epoch = info.train.epoch + 1
	end

	local context = {}
	local max_epochs = info.options.max_epochs or 1000
	context.params, context.grad_params = info.model.model:getParameters()
	context.confusion = optim.ConfusionMatrix(10)

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
