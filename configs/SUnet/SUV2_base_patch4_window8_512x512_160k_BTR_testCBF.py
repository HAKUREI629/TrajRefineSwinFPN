_base_ = [
    '../_base_/models/S_unet_v2.py', '../_base_/datasets/BTR_testCBF512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        img_size=512,
        patch_size=4,
        in_chans=3,
        out_chans=256,
        embed_dim=128,
        depths=[8, 8, 8, 8],
        num_heads=[8, 8, 8, 8],
        window_size=8,
    ),
    decode_head=dict(num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4)
