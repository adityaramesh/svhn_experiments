require "source/utilities/csv_logger"

local logger = CSVLogger.create("test", {"iteration", "stuff"})
logger:log_value("iteration", 1)
logger:log_array("stuff", {1, 2, 3, 4})
logger:flush()
logger:close()
