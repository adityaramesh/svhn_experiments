require "lfs"
require "torch"

local model_dir = "models/batch_100"
for file in lfs.dir(model_dir) do
	local path = paths.concat(model_dir, file)
	local info_file = paths.concat(path, "acc_info.t7")
	if paths.filep(info_file) then
		local info = torch.load(info_file)
		print(file, #info.train_scores)
	end
end
