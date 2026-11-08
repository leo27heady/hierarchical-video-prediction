method = 'HVPIntegrated'

# 256 flat fine candidates versus 64 coarse + 16 detail tokens: 3.2x.
d_model = 128
stem_width = 48
core_depth = 6
decoder_depth = 2
num_heads = 4
mlp_ratio = 4
hierarchy_mode = 'hierarchical'
routing_mode = 'mixed'
detail_quota = 16
min_detail_per_frame = 1
mixed_dynamic_probability = 0.75
density_weight = 1.0
change_weight = 1.0
learned_weight = 0.5
reversible_core = True
position_frequencies = 6
validation_fraction = 0.1
validation_gap = 8
memmap_cache = 'taxibj/memmap_v1'
strict_memmap = True

# TaxiBJ is smooth and short; keep the official OpenSTL-style cosine schedule.
lr = 5e-4
batch_size = 16
val_batch_size = 16
epoch = 100
sched = 'cosine'
drop_path = 0.0
warmup_epoch = 5
clip_grad = 1.0
