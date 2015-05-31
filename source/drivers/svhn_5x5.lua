--
-- The baseline converges to 90.53% validation accuracy.
--

require "source/models/cnn_5x5"
require "source/utilities/run_model_verbose"

run(get_model_info, get_train_info)
