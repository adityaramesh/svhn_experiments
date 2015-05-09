--
-- Creates a subset of the training data.
--

require "torch"

local raw_data_dir = "data/raw/"
local train = torch.load(raw_data_dir .. "train.t7")

assert(train.inputs:size(1) == 73257)
assert(train.inputs:size(2) == 3)
assert(train.inputs:size(3) == 32)
assert(train.inputs:size(4) == 32)
assert(train.targets:size(1) == 73257)

local subset_size = 10000
assert(subset_size <= train.inputs:size(1))

local inputs = torch.ByteTensor(subset_size, train.inputs:size(2),
	train.inputs:size(3), train.inputs:size(4))
local targets = torch.DoubleTensor(subset_size)
local perm = torch.randperm(train.inputs:size(1))

for i = 1, subset_size do
	inputs[i]:copy(train.inputs[perm[i]])
	targets[i] = train.targets[perm[i]]
end

local new_data = {inputs = inputs, targets = targets}
torch.save(raw_data_dir .. "train_small.t7", new_data)
