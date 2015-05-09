--
-- Converts the original SVHN data from mat to t7 format.
--

require "torch"
require "mattorch"

local function convert_to_t7(data)
	return {
		inputs = data.X:transpose(3, 4),
		targets = data.y[1]
	}
end

local raw_data_dir = "data/raw/"
local train = mattorch.load(raw_data_dir .. "train_32x32.mat")
local test = mattorch.load(raw_data_dir .. "test_32x32.mat")
torch.save(raw_data_dir .. "train.t7", convert_to_t7(train))
torch.save(raw_data_dir .. "test.t7", convert_to_t7(test))
