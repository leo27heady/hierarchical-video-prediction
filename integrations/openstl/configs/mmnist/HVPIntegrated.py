method = 'HVPIntegrated'

# Integrated model: 2,560 flat fine candidates versus 640 coarse + 160 detail
# tokens gives an analytic 3.2x retained-token ratio before measured CUDA peaks.
d_model = 128
stem_width = 48
core_depth = 6
decoder_depth = 2
num_heads = 4
mlp_ratio = 4
hierarchy_mode = 'hierarchical'
routing_mode = 'mixed'
detail_quota = 160
min_detail_per_frame = 4
mixed_dynamic_probability = 0.75
density_weight = 1.0
change_weight = 1.0
learned_weight = 0.5
reversible_core = True
position_frequencies = 6
validation_file = 'moving_mnist/mnist_validation_seq_seed260902.npy'
strict_validation = True

# Long-run schedule is provisional until the bounded speed/convergence gate.
lr = 5e-4
batch_size = 8
val_batch_size = 8
epoch = 200
sched = 'onecycle'
drop_path = 0.0
warmup_epoch = 0
clip_grad = 1.0
