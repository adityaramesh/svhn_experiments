require "lfs"
require "torch"

local model_dir = "models/batch_100"
for file in lfs.dir(model_dir) do
	local path = paths.concat(model_dir, file)
	local info_file = paths.concat(path, "acc_info.t7")
	print("Processing model \"" .. file .. "\".")

	if paths.filep(info_file) then
		local info = torch.load(info_file)
		local new_scores = {}
		for k, v in pairs(info.test_scores) do
			new_scores[k - 1] = v
		end

		info.test_scores = new_scores
		torch.save(info_file, info)
	end
end
