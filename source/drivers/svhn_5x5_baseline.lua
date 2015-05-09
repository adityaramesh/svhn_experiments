--
-- The baseline converges to 90.53% validation accuracy.
--

require "source/models/cnn_5x5"
require "source/utilities/run_model"

-- These optimization algorithms have been verified to work using their current
-- settings.
require "source/train/adadelta_100"
--require "source/train/adam_100"
--require "source/train/rmsprop_100"

-- These optimization algorithms have been verified to work:
-- require "source/train/sgu_100"

run(get_model_info, get_train_info)
