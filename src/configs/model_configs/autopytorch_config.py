
# Default config for autopytorch model
# Can be overwritten in the config file of a model

autopytorch_config = {
    "transform__0__method_name": "gaussienize",
    "transform__0__type": "quantile",
    "transform__0__apply_on": "numerical",
}

autopytorch_config_default = {}

# autopytorch_config_default["model__batch_size"] = {"value": 512}
# autopytorch_config_default["transformed_target"] = {"value": True}
# Configuration:
#   data_loader:batch_size, Constant: 128
#   encoder:__choice__, Value: 'NoEncoder'
#   feature_preprocessor:__choice__, Value: 'NoFeaturePreprocessor'
#   imputer:numerical_strategy, Value: 'median'
#   lr_scheduler:CosineAnnealingWarmRestarts:n_restarts, Constant: 3
#   lr_scheduler:__choice__, Value: 'CosineAnnealingWarmRestarts'
#   network_backbone:ShapedResNetBackbone:activation, Value: 'relu'
#   network_backbone:ShapedResNetBackbone:blocks_per_group, Constant: 2
#   network_backbone:ShapedResNetBackbone:max_shake_drop_probability, Value: 0.5
#   network_backbone:ShapedResNetBackbone:max_units, Constant: 512
#   network_backbone:ShapedResNetBackbone:multi_branch_choice, Value: 'shake-drop'
#   network_backbone:ShapedResNetBackbone:num_groups, Constant: 2
#   network_backbone:ShapedResNetBackbone:output_dim, Constant: 512
#   network_backbone:ShapedResNetBackbone:resnet_shape, Value: 'brick'
#   network_backbone:ShapedResNetBackbone:shake_shake_update_func, Value: 'even-even'
#   network_backbone:ShapedResNetBackbone:use_batch_norm, Value: False
#   network_backbone:ShapedResNetBackbone:use_dropout, Value: False
#   network_backbone:ShapedResNetBackbone:use_skip_connection, Value: True
#   network_backbone:__choice__, Value: 'ShapedResNetBackbone'
#   network_embedding:__choice__, Value: 'NoEmbedding'
#   network_head:__choice__, Value: 'no_head'
#   network_head:no_head:activation, Value: 'relu'
#   network_init:NoInit:bias_strategy, Value: 'Normal'
#   network_init:__choice__, Value: 'NoInit'
#   optimizer:AdamWOptimizer:beta1, Constant: 0.9
#   optimizer:AdamWOptimizer:beta2, Constant: 0.999
#   optimizer:AdamWOptimizer:lr, Constant: 0.001
#   optimizer:AdamWOptimizer:use_weight_decay, Value: True
#   optimizer:AdamWOptimizer:weight_decay, Value: 0.0001
#   optimizer:__choice__, Value: 'AdamWOptimizer'
#   scaler:__choice__, Value: 'StandardScaler'
#   trainer:StandardTrainer:Lookahead:la_alpha, Value: 0.6
#   trainer:StandardTrainer:Lookahead:la_steps, Value: 6
#   trainer:StandardTrainer:se_lastk, Constant: 3
#   trainer:StandardTrainer:use_lookahead_optimizer, Value: True
#   trainer:StandardTrainer:use_snapshot_ensemble, Value: True
#   trainer:StandardTrainer:use_stochastic_weight_averaging, Value: True
#   trainer:StandardTrainer:weighted_loss, Constant: 1
#   trainer:__choice__, Value: 'StandardTrainer'


# Configuration space object:
#   Hyperparameters:
#     data_loader:batch_size, Type: Constant, Value: 128
#     encoder:__choice__, Type: Categorical, Choices: {NoEncoder}, Default: NoEncoder
#     feature_preprocessor:__choice__, Type: Categorical, Choices: {NoFeaturePreprocessor}, Default: NoFeaturePreprocessor
#     imputer:numerical_strategy, Type: Categorical, Choices: {median}, Default: median
#     lr_scheduler:CosineAnnealingWarmRestarts:n_restarts, Type: Constant, Value: 3
#     lr_scheduler:__choice__, Type: Categorical, Choices: {CosineAnnealingWarmRestarts}, Default: CosineAnnealingWarmRestarts
#     network_backbone:ShapedResNetBackbone:activation, Type: Categorical, Choices: {relu}, Default: relu
#     network_backbone:ShapedResNetBackbone:blocks_per_group, Type: Constant, Value: 2
#     network_backbone:ShapedResNetBackbone:dropout_shape, Type: Categorical, Choices: {funnel, long_funnel, diamond, hexagon, brick, triangle, stairs}, Default: funnel
#     network_backbone:ShapedResNetBackbone:max_dropout, Type: UniformFloat, Range: [0.0, 0.8], Default: 0.5
#     network_backbone:ShapedResNetBackbone:max_shake_drop_probability, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5
#     network_backbone:ShapedResNetBackbone:max_units, Type: Constant, Value: 512
#     network_backbone:ShapedResNetBackbone:multi_branch_choice, Type: Categorical, Choices: {shake-drop, shake-shake, None}, Default: shake-drop
#     network_backbone:ShapedResNetBackbone:num_groups, Type: Constant, Value: 2
#     network_backbone:ShapedResNetBackbone:output_dim, Type: Constant, Value: 512
#     network_backbone:ShapedResNetBackbone:resnet_shape, Type: Categorical, Choices: {brick}, Default: brick
#     network_backbone:ShapedResNetBackbone:shake_shake_update_func, Type: Categorical, Choices: {even-even}, Default: even-even
#     network_backbone:ShapedResNetBackbone:use_batch_norm, Type: Categorical, Choices: {True, False}, Default: False
#     network_backbone:ShapedResNetBackbone:use_dropout, Type: Categorical, Choices: {True, False}, Default: False
#     network_backbone:ShapedResNetBackbone:use_skip_connection, Type: Categorical, Choices: {True, False}, Default: True
#     network_backbone:__choice__, Type: Categorical, Choices: {ShapedResNetBackbone}, Default: ShapedResNetBackbone
#     network_embedding:__choice__, Type: Categorical, Choices: {NoEmbedding}, Default: NoEmbedding
#     network_head:__choice__, Type: Categorical, Choices: {no_head}, Default: no_head
#     network_head:no_head:activation, Type: Categorical, Choices: {relu}, Default: relu
#     network_init:NoInit:bias_strategy, Type: Categorical, Choices: {Zero, Normal}, Default: Normal
#     network_init:__choice__, Type: Categorical, Choices: {NoInit}, Default: NoInit
#     optimizer:AdamWOptimizer:beta1, Type: Constant, Value: 0.9
#     optimizer:AdamWOptimizer:beta2, Type: Constant, Value: 0.999
#     optimizer:AdamWOptimizer:lr, Type: Constant, Value: 0.001
#     optimizer:AdamWOptimizer:use_weight_decay, Type: Categorical, Choices: {True, False}, Default: True
#     optimizer:AdamWOptimizer:weight_decay, Type: UniformFloat, Range: [1e-05, 0.1], Default: 0.0001
#     optimizer:__choice__, Type: Categorical, Choices: {AdamWOptimizer}, Default: AdamWOptimizer
#     scaler:__choice__, Type: Categorical, Choices: {StandardScaler}, Default: StandardScaler
#     trainer:AdversarialTrainer:Lookahead:la_alpha, Type: UniformFloat, Range: [0.5, 0.8], Default: 0.6
#     trainer:AdversarialTrainer:Lookahead:la_steps, Type: UniformInteger, Range: [5, 10], Default: 6
#     trainer:AdversarialTrainer:epsilon, Type: Constant, Value: 0.007
#     trainer:AdversarialTrainer:se_lastk, Type: Constant, Value: 3
#     trainer:AdversarialTrainer:use_lookahead_optimizer, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:AdversarialTrainer:use_snapshot_ensemble, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:AdversarialTrainer:use_stochastic_weight_averaging, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:AdversarialTrainer:weighted_loss, Type: Constant, Value: 1
#     trainer:MixUpTrainer:Lookahead:la_alpha, Type: UniformFloat, Range: [0.5, 0.8], Default: 0.6
#     trainer:MixUpTrainer:Lookahead:la_steps, Type: UniformInteger, Range: [5, 10], Default: 6
#     trainer:MixUpTrainer:alpha, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.2
#     trainer:MixUpTrainer:se_lastk, Type: Constant, Value: 3
#     trainer:MixUpTrainer:use_lookahead_optimizer, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:MixUpTrainer:use_snapshot_ensemble, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:MixUpTrainer:use_stochastic_weight_averaging, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:MixUpTrainer:weighted_loss, Type: Constant, Value: 1
#     trainer:RowCutMixTrainer:Lookahead:la_alpha, Type: UniformFloat, Range: [0.5, 0.8], Default: 0.6
#     trainer:RowCutMixTrainer:Lookahead:la_steps, Type: UniformInteger, Range: [5, 10], Default: 6
#     trainer:RowCutMixTrainer:alpha, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.2
#     trainer:RowCutMixTrainer:se_lastk, Type: Constant, Value: 3
#     trainer:RowCutMixTrainer:use_lookahead_optimizer, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:RowCutMixTrainer:use_snapshot_ensemble, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:RowCutMixTrainer:use_stochastic_weight_averaging, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:RowCutMixTrainer:weighted_loss, Type: Constant, Value: 1
#     trainer:RowCutOutTrainer:Lookahead:la_alpha, Type: UniformFloat, Range: [0.5, 0.8], Default: 0.6
#     trainer:RowCutOutTrainer:Lookahead:la_steps, Type: UniformInteger, Range: [5, 10], Default: 6
#     trainer:RowCutOutTrainer:cutout_prob, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.2
#     trainer:RowCutOutTrainer:patch_ratio, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.2
#     trainer:RowCutOutTrainer:se_lastk, Type: Constant, Value: 3
#     trainer:RowCutOutTrainer:use_lookahead_optimizer, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:RowCutOutTrainer:use_snapshot_ensemble, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:RowCutOutTrainer:use_stochastic_weight_averaging, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:RowCutOutTrainer:weighted_loss, Type: Constant, Value: 1
#     trainer:StandardTrainer:Lookahead:la_alpha, Type: UniformFloat, Range: [0.5, 0.8], Default: 0.6
#     trainer:StandardTrainer:Lookahead:la_steps, Type: UniformInteger, Range: [5, 10], Default: 6
#     trainer:StandardTrainer:se_lastk, Type: Constant, Value: 3
#     trainer:StandardTrainer:use_lookahead_optimizer, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:StandardTrainer:use_snapshot_ensemble, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:StandardTrainer:use_stochastic_weight_averaging, Type: Categorical, Choices: {True, False}, Default: True
#     trainer:StandardTrainer:weighted_loss, Type: Constant, Value: 1
#     trainer:__choice__, Type: Categorical, Choices: {AdversarialTrainer, MixUpTrainer, RowCutMixTrainer, RowCutOutTrainer, StandardTrainer}, Default: StandardTrainer
#   Conditions:
#     lr_scheduler:CosineAnnealingWarmRestarts:n_restarts | lr_scheduler:__choice__ == 'CosineAnnealingWarmRestarts'
#     network_backbone:ShapedResNetBackbone:activation | network_backbone:__choice__ == 'ShapedResNetBackbone'
#     network_backbone:ShapedResNetBackbone:blocks_per_group | network_backbone:__choice__ == 'ShapedResNetBackbone'
#     network_backbone:ShapedResNetBackbone:dropout_shape | network_backbone:ShapedResNetBackbone:use_dropout == True
#     network_backbone:ShapedResNetBackbone:max_dropout | network_backbone:ShapedResNetBackbone:use_dropout == True
#     network_backbone:ShapedResNetBackbone:max_shake_drop_probability | network_backbone:ShapedResNetBackbone:multi_branch_choice == 'shake-drop'
#     network_backbone:ShapedResNetBackbone:max_units | network_backbone:__choice__ == 'ShapedResNetBackbone'
#     network_backbone:ShapedResNetBackbone:multi_branch_choice | network_backbone:ShapedResNetBackbone:use_skip_connection == True
#     network_backbone:ShapedResNetBackbone:num_groups | network_backbone:__choice__ == 'ShapedResNetBackbone'
#     network_backbone:ShapedResNetBackbone:output_dim | network_backbone:__choice__ == 'ShapedResNetBackbone'
#     network_backbone:ShapedResNetBackbone:resnet_shape | network_backbone:__choice__ == 'ShapedResNetBackbone'
#     network_backbone:ShapedResNetBackbone:shake_shake_update_func | network_backbone:ShapedResNetBackbone:multi_branch_choice in {'shake-drop', 'shake-shake'}
#     network_backbone:ShapedResNetBackbone:use_batch_norm | network_backbone:__choice__ == 'ShapedResNetBackbone'
#     network_backbone:ShapedResNetBackbone:use_dropout | network_backbone:__choice__ == 'ShapedResNetBackbone'
#     network_backbone:ShapedResNetBackbone:use_skip_connection | network_backbone:__choice__ == 'ShapedResNetBackbone'
#     network_head:no_head:activation | network_head:__choice__ == 'no_head'
#     network_init:NoInit:bias_strategy | network_init:__choice__ == 'NoInit'
#     optimizer:AdamWOptimizer:beta1 | optimizer:__choice__ == 'AdamWOptimizer'
#     optimizer:AdamWOptimizer:beta2 | optimizer:__choice__ == 'AdamWOptimizer'
#     optimizer:AdamWOptimizer:lr | optimizer:__choice__ == 'AdamWOptimizer'
#     optimizer:AdamWOptimizer:use_weight_decay | optimizer:__choice__ == 'AdamWOptimizer'
#     optimizer:AdamWOptimizer:weight_decay | optimizer:AdamWOptimizer:use_weight_decay == True
#     trainer:AdversarialTrainer:Lookahead:la_alpha | trainer:AdversarialTrainer:use_lookahead_optimizer == True
#     trainer:AdversarialTrainer:Lookahead:la_steps | trainer:AdversarialTrainer:use_lookahead_optimizer == True
#     trainer:AdversarialTrainer:epsilon | trainer:__choice__ == 'AdversarialTrainer'
#     trainer:AdversarialTrainer:se_lastk | trainer:AdversarialTrainer:use_snapshot_ensemble == True
#     trainer:AdversarialTrainer:use_lookahead_optimizer | trainer:__choice__ == 'AdversarialTrainer'
#     trainer:AdversarialTrainer:use_snapshot_ensemble | trainer:__choice__ == 'AdversarialTrainer'
#     trainer:AdversarialTrainer:use_stochastic_weight_averaging | trainer:__choice__ == 'AdversarialTrainer'
#     trainer:AdversarialTrainer:weighted_loss | trainer:__choice__ == 'AdversarialTrainer'
#     trainer:MixUpTrainer:Lookahead:la_alpha | trainer:MixUpTrainer:use_lookahead_optimizer == True
#     trainer:MixUpTrainer:Lookahead:la_steps | trainer:MixUpTrainer:use_lookahead_optimizer == True
#     trainer:MixUpTrainer:alpha | trainer:__choice__ == 'MixUpTrainer'
#     trainer:MixUpTrainer:se_lastk | trainer:MixUpTrainer:use_snapshot_ensemble == True
#     trainer:MixUpTrainer:use_lookahead_optimizer | trainer:__choice__ == 'MixUpTrainer'
#     trainer:MixUpTrainer:use_snapshot_ensemble | trainer:__choice__ == 'MixUpTrainer'
#     trainer:MixUpTrainer:use_stochastic_weight_averaging | trainer:__choice__ == 'MixUpTrainer'
#     trainer:MixUpTrainer:weighted_loss | trainer:__choice__ == 'MixUpTrainer'
#     trainer:RowCutMixTrainer:Lookahead:la_alpha | trainer:RowCutMixTrainer:use_lookahead_optimizer == True
#     trainer:RowCutMixTrainer:Lookahead:la_steps | trainer:RowCutMixTrainer:use_lookahead_optimizer == True
#     trainer:RowCutMixTrainer:alpha | trainer:__choice__ == 'RowCutMixTrainer'
#     trainer:RowCutMixTrainer:se_lastk | trainer:RowCutMixTrainer:use_snapshot_ensemble == True
#     trainer:RowCutMixTrainer:use_lookahead_optimizer | trainer:__choice__ == 'RowCutMixTrainer'
#     trainer:RowCutMixTrainer:use_snapshot_ensemble | trainer:__choice__ == 'RowCutMixTrainer'
#     trainer:RowCutMixTrainer:use_stochastic_weight_averaging | trainer:__choice__ == 'RowCutMixTrainer'
#     trainer:RowCutMixTrainer:weighted_loss | trainer:__choice__ == 'RowCutMixTrainer'
#     trainer:RowCutOutTrainer:Lookahead:la_alpha | trainer:RowCutOutTrainer:use_lookahead_optimizer == True
#     trainer:RowCutOutTrainer:Lookahead:la_steps | trainer:RowCutOutTrainer:use_lookahead_optimizer == True
#     trainer:RowCutOutTrainer:cutout_prob | trainer:__choice__ == 'RowCutOutTrainer'
#     trainer:RowCutOutTrainer:patch_ratio | trainer:__choice__ == 'RowCutOutTrainer'
#     trainer:RowCutOutTrainer:se_lastk | trainer:RowCutOutTrainer:use_snapshot_ensemble == True
#     trainer:RowCutOutTrainer:use_lookahead_optimizer | trainer:__choice__ == 'RowCutOutTrainer'
#     trainer:RowCutOutTrainer:use_snapshot_ensemble | trainer:__choice__ == 'RowCutOutTrainer'
#     trainer:RowCutOutTrainer:use_stochastic_weight_averaging | trainer:__choice__ == 'RowCutOutTrainer'
#     trainer:RowCutOutTrainer:weighted_loss | trainer:__choice__ == 'RowCutOutTrainer'
#     trainer:StandardTrainer:Lookahead:la_alpha | trainer:StandardTrainer:use_lookahead_optimizer == True
#     trainer:StandardTrainer:Lookahead:la_steps | trainer:StandardTrainer:use_lookahead_optimizer == True
#     trainer:StandardTrainer:se_lastk | trainer:StandardTrainer:use_snapshot_ensemble == True
#     trainer:StandardTrainer:use_lookahead_optimizer | trainer:__choice__ == 'StandardTrainer'
#     trainer:StandardTrainer:use_snapshot_ensemble | trainer:__choice__ == 'StandardTrainer'
#     trainer:StandardTrainer:use_stochastic_weight_averaging | trainer:__choice__ == 'StandardTrainer'
#     trainer:StandardTrainer:weighted_loss | trainer:__choice__ == 'StandardTrainer'