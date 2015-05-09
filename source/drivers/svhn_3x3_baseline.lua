--
-- The baseline should converge to over 90% validation accuracy. I did not wait
-- until convergene. Note: I could not get SGU or RMSProp to work for this
-- model. Perhaps they might with better hyperparameter selection.
--

require "source/models/cnn_3x3"
require "source/utilities/run_model"

-- These optimization algorithms have been verified to work using their current
-- settings.
require "source/train/adadelta_100"
--require "source/train/rmsprop_100"
--require "source/train/adam_100"

run(get_model_info, get_train_info)
