require "torch"

CSVLogger = {}
CSVLogger.__index = CSVLogger

function CSVLogger.create(model_name, fields)
	local self = {}
	setmetatable(self, CSVLogger)

	local log_dir = "logs"
	local log_path = paths.concat(log_dir, model_name .. "_output.log")
	self.file = io.open(log_path, "w")

	self.fields = {}
	self.indices = {}
	self.cur_row = {}
	for k = 1, #fields do
		self.indices[fields[k]] = k
		self.cur_row[k] = "\"\""
	end

	self.file:write(table.concat(fields, ", "), "\n")
	return self
end

function CSVLogger:log_value(field, value)
	assert(self.indices[field] ~= nil)

	local index = self.indices[field]
	self.cur_row[index] = tostring(value)
end

function CSVLogger:log_array(field, values)
	assert(self.indices[field] ~= nil)

	local index = self.indices[field]
	self.cur_row[index] = "\"" .. table.concat(values, ", ") .. "\""
end

function CSVLogger:flush()
	self.file:write(table.concat(self.cur_row, ", "), "\n")
	self.file:flush()

	for k = 1, #self.cur_row do
		self.cur_row[k] = "\"\""
	end
end

function CSVLogger:close()
	self.file:close()
end
