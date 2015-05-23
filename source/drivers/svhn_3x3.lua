--
-- The baseline should converge to over 90% validation accuracy. I did not wait
-- until convergence.
--
-- Note: I could not get SGU or RMSProp to work for this model. Perhaps they
-- might with better hyperparameter selection.
--

require "source/models/cnn_3x3"
require "source/utilities/run_model"

run(get_model_info, get_train_info)
