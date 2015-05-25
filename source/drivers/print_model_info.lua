require "lfs"
require "torch"

local model_dir = "models/svhn_3x3_batch_100"
for file in lfs.dir(model_dir) do
	local path = paths.concat(model_dir, file)
	local info_file = paths.concat(path, "acc_info.t7")
	if paths.filep(info_file) then
		local info = torch.load(info_file)
		io.write(file .. " " .. info.best_test .. "\n")
	end
end
