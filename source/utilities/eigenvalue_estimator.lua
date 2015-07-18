require "torch"

--
-- Based on Xiang's code here:
-- https://github.com/zhangxiangxiao/GalaxyZoo/blob/master/model.lua
--
local function get_dropout_prob(model)
	for i, m in ipairs(model.modules) do
		if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
			return m.p
		end
	end
end

--
-- Taken from Xiang's code here:
-- https://github.com/zhangxiangxiao/GalaxyZoo/blob/master/model.lua
--
local function change_dropout_prob(model, prob)
	for i, m in ipairs(model.modules) do
		if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
			m.p = prob
		end
	end
end

local function isnan(x)
	return x ~= x
end

local function grid_search(eig_func, input, target, eps, tol, gran)
	assert(eps > 0 and eps < 1)
	assert(gran > 0 and gran < 1)

	local eig = 0
	local best_skew = 1
	local best_eps = 0

	local a = math.max(eps - 10 * gran, gran)
	local b = eps + 10 * gran

	for cur_eps = a, b + gran, gran do
		local cur_eig, cur_skew = eig_func(input, target, cur_eps)

		if cur_skew < best_skew then
			if cur_skew < tol then
				return cur_eig, cur_skew, cur_eps
			end

			eig = cur_eig
			best_skew = cur_skew
			best_eps = cur_eps
		end
	end

	return eig, best_skew, best_eps
end

local function iterative_grid_search(eig_func, input, target, skew, eps, tol, max_refinements)
	assert(eps > 0 and eps < 1)
	assert(tol > 0 and tol < 1)

	local exp = math.floor(math.log(eps) / math.log(10))
	-- Controls the granularity of the grid search at each refinement.
	local gran = math.pow(10, exp - 1)
	local best_skew = skew
	local best_eps = eps
	local eig = 0

	for i = 1, max_refinements do
		local cur_eig, cur_skew, cur_eps = grid_search(eig_func, input,
			target, best_eps, tol, gran)

		if cur_skew < best_skew then
			if cur_skew < tol then
				return cur_eig, cur_skew, cur_eps
			end

			eig = cur_eig
			best_skew = cur_skew
			best_eps = cur_eps
		end
		gran = gran / 10
	end

	return eig, best_skew, best_eps
end

-- Note: we might get this to run faster by giving up and reinitializing
-- `init_phi` sooner if we aren't able to find a satisfactory value for `eps`.
local function find_eps(eig_func, input, target, eps_list, tol)
	local eig = 0
	local best_skew = 1
	local best_eps = 0
	local results = {}

	-- First try all of the `eps` values that are in `eps_list` for the
	-- finite difference.
	for j = 1, #eps_list do
		local cur_eps = eps_list[j]
		local cur_eig, cur_skew = eig_func(input, target, cur_eps)

		if not isnan(cur_skew) then
			if cur_skew < best_skew then
				if cur_skew < tol then
					return cur_eig, cur_skew, cur_eps
				end

				eig = cur_eig
				best_skew = cur_skew
				best_eps = cur_eps
			end
			results[cur_skew] = cur_eps
		end
	end

	-- If we get here, then we weren't able to find a satisfactory value for
	-- `eps`. To try to hone in on a better value, we run a refined grid
	-- search over the top five results.
	local sorted_results = {}
	for result in pairs(results) do
		table.insert(sorted_results, result)
	end
	table.sort(sorted_results)

	for j = 1, math.min(#sorted_results, 5) do
		local cur_skew = sorted_results[j]
		local cur_eps = results[cur_skew]
		local new_eig, new_skew, new_eps = iterative_grid_search(
			eig_func, input, target, cur_skew, cur_eps, tol, 3)

		if new_skew < best_skew then
			if new_skew < tol then
				return new_eig, new_skew, new_eps
			end

			eig = new_eig
			best_skew = new_skew
			best_eps = new_eps
		end
	end

	-- If we still weren't able to find a satisfactory value for `eps`, then
	-- we return the best one found.
	return eig, best_skew, best_eps
end

EigenvalueEstimator = {}
EigenvalueEstimator.__index = EigenvalueEstimator

--
-- Warning: the code currently assumes that (1) the model is of type
-- `nn.Sequential` and (2) all dropout modules in the model use the same dropout
-- probability.
--
function EigenvalueEstimator.create(model, params, grad_params, grad_func)
	local self = {}
	setmetatable(self, EigenvalueEstimator)

	self.model        = model
	self.params       = params
	self.grad_params  = grad_params
	self.grad_func    = grad_func
	self.dropout_prob = get_dropout_prob(model)

	self.phi      = torch.Tensor():typeAs(params):resizeAs(params)
	self.init_phi = torch.randn(params:size(1)):typeAs(params)
	self.init_phi:div(self.init_phi:norm())

	-- Used to store the eigenvector associated with the maximum magnitude
	-- eigenvalue when computing the minimum and maximum eigenvalues.
	self.tmp_phi    = torch.Tensor():typeAs(params):resizeAs(params)
	self.tmp_grad   = torch.Tensor():typeAs(params):resizeAs(params)
	self.tmp_params = torch.Tensor():typeAs(params):resizeAs(params)

	-- Number of iterations of the power method.
	self.max_power_iters = 10

	-- Maximum acceptable tolerance for angle between the eigenvector and
	-- its product with the Hessian. The current value is actually quite
	-- forgiving. If the tolerance is much larger than 1e-8, then the
	-- corresponding eigenvalue will only be correct to within an order of
	-- magnitude. I wouldn't recommend making this any smaller.
	self.eigvec_skew_tol = 1e-6

	-- List of trial values for epsilon, the constant used when computing
	-- finite differences.
	self.eps_list = {
		-- For some reason, 6.56e-8 seems to give good results for a
		-- large fraction of the finite differences.
		6.56e-8,
		1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8, 9e-8,
		1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7,
		1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9, 9e-9,
		1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6,
		1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5
	}
	return self
end

function EigenvalueEstimator:compute_max_mag_eig(input, target, eps)
	-- "Hot-starting" the iterations using the previous value of `phi`
	-- actually seems to retard progress. Instead, we use the saved value of
	-- `init_phi`.
	self.phi:copy(self.init_phi)

	local proj = 0
	local skew = 0
	local norm = 0

	for j = 1, self.max_power_iters do
		-- Because of finite precision arithmetic, we can't undo the
		-- action of adding `-eps * phi` to `params` by subtracting the
		-- same quantity. Instead, we save `params` and restore it after
		-- bprop.
		self.params:add(-eps, self.phi)
		self.grad_func(input, target)
		self.tmp_grad:copy(self.grad_params)

		self.params:copy(self.tmp_params)
		self.params:add(eps, self.phi)
		self.grad_func(input, target)
		self.params:copy(self.tmp_params)
		self.grad_params:add(-1, self.tmp_grad):div(2 * eps)

		-- How close is `phi` to being an eigenvector?
		norm = self.grad_params:norm()
		proj = self.phi:dot(self.grad_params) / norm
		skew = math.min(math.abs(proj - 1), math.abs(proj + 1))
		if skew < self.eigvec_skew_tol then break end

		self.phi:copy(self.grad_params)
		self.phi:div(norm)
	end

	-- Note: we don't return the eigenvector, because it is already a member
	-- variable of this class.
	norm = proj > 0 and norm or -norm
	return norm, skew
end

--
-- Given the eigenvalue of the largest magnitude, computes either the largest
-- positive eigenvalue or the largest negative eigenvalue (depending on the sign
-- of the given eigenvalue). This is done by applying the power method to `(H -
-- lambda I)`, where `lambda` is the given eigenvalue. As before, the
-- Hessian-vector products are computed using finite differences.
--
function EigenvalueEstimator:compute_extreme_eig(input, target, max_mag_eig, eps)
	-- "Hot-starting" the iterations using the previous value of `phi`
	-- actually seems to retard progress. Instead, we use the saved value of
	-- `init_phi`.
	self.phi:copy(self.init_phi)

	local proj = 0
	local skew = 0
	local norm = 0

	for j = 1, self.max_power_iters do
		-- Because of finite precision arithmetic, we can't undo the
		-- action of adding `-eps * phi` to `params` by subtracting the
		-- same quantity. Instead, we save `params` and restore it after
		-- bprop.
		self.params:add(-eps, self.phi)
		self.grad_func(input, target)
		self.tmp_grad:copy(self.grad_params)

		self.params:copy(self.tmp_params)
		self.params:add(eps, self.phi)
		self.grad_func(input, target)
		self.params:copy(self.tmp_params)
		self.grad_params:add(-1, self.tmp_grad):div(2 * eps)
		self.grad_params:add(-max_mag_eig, self.phi)

		-- How close is `phi` to being an eigenvector?
		norm = self.grad_params:norm()
		proj = self.phi:dot(self.grad_params) / norm
		skew = math.min(math.abs(proj - 1), math.abs(proj + 1))
		if skew < self.eigvec_skew_tol then break end

		self.phi:copy(self.grad_params)
		self.phi:div(norm)
	end

	-- Note: we don't return the eigenvector, because it is already a member
	-- variable of this class.
	norm = proj > 0 and norm or -norm
	return norm + max_mag_eig, skew
end

function EigenvalueEstimator:check_eig(input, target, eps)
	self.params:add(-eps, self.phi)
	self.grad_func(input, target)
	self.tmp_grad:copy(self.grad_params)

	self.params:copy(self.tmp_params)
	self.params:add(eps, self.phi)
	self.grad_func(input, target)
	self.params:copy(self.tmp_params)
	self.grad_params:add(-1, self.tmp_grad):div(2 * eps)

	-- How close is `phi` to being an eigenvector?
	local norm = self.grad_params:norm()
	local proj = self.phi:dot(self.grad_params) / norm
	local skew = math.min(math.abs(proj - 1), math.abs(proj + 1))

	-- Note: we don't return the eigenvector, because it is already a member
	-- variable of this class.
	norm = proj > 0 and norm or -norm
	return norm, skew
end

--
-- Caution: after this function returns, the gradient parameters of the model
-- will be in an undefined state. If you wish to preserve the value of the
-- gradient prior to calling this function, then you will need to save it
-- explicitly. However, the parameters of the model are guaranteed to be
-- preserved.
--
function EigenvalueEstimator:compute_eig(input, target, eig_func)
	local eig, best_skew, best_eps = find_eps(eig_func, input, target,
		self.eps_list, self.eigvec_skew_tol)

	if best_skew < self.eigvec_skew_tol then
		local loss = self.grad_func(input, target)
		return eig, self.phi, loss, self.grad_params:norm()
	else
		-- If we still weren't able to find a satisfactory value for
		-- `eps` after the refined grid search, then reinitialize
		-- `init_phi` and try again.
		for j = 1, 5 do
			self.init_phi:copy(torch.randn(self.params:size(1)))
			self.init_phi:div(self.init_phi:norm())

			local cur_eig, cur_skew, cur_eps =
				find_eps(eig_func, input, target, self.eps_list,
				self.eigvec_skew_tol)

			if cur_skew < best_skew then
				if cur_skew < self.eigvec_skew_tol then
					change_dropout_prob(self.model, self.dropout_prob)
					local loss = self.grad_func(input, target)
					return cur_eig, self.phi, loss, self.grad_params:norm()
				end

				eig = cur_eig
				best_skew = cur_skew
				best_eps = cur_eps
			end
		end
	end

	print("Failed to compute eigenvalue to satisfactory accuracy.")
	print("Best skew: " .. best_skew .. ".")
	print("Best epsilon: " .. best_eps .. ".")
end

function EigenvalueEstimator:get_max_mag_eig(input, target)
	self.tmp_params:copy(self.params)
	local eig_func = function(input, target, eps)
		return self:compute_max_mag_eig(input, target, eps)
	end

	change_dropout_prob(self.model, 0)
	local eig, _, loss, norm_grad = self:compute_eig(input, target, eig_func)
	change_dropout_prob(self.model, self.dropout_prob)
	return eig, self.phi, loss, norm_grad
end

function EigenvalueEstimator:get_min_max_eig(input, target)
	self.tmp_params:copy(self.params)
	local eig_func_1 = function(input, target, eps)
		return self:compute_max_mag_eig(input, target, eps)
	end

	change_dropout_prob(self.model, 0)
	local eig_1, _, _, _ = self:compute_eig(input, target, eig_func_1)

	if not eig_1 then
		change_dropout_prob(self.model, self.dropout_prob)
		return
	end

	local eig_func_2 = function(input, target, eps)
		return self:compute_extreme_eig(input, target, eig_1, eps)
	end

	self.tmp_phi:copy(self.phi)
	local eig_2, _, _, _ = self:compute_eig(input, target, eig_func_2)
	change_dropout_prob(self.model, self.dropout_prob)

	if not eig_2 then
		return eig_1, self.tmp_phi
	end

	if eig_1 < eig_2 then
		return eig_1, self.tmp_phi, eig_2, self.phi
	else
		return eig_2, self.phi, eig_1, self.tmp_phi
	end
end
