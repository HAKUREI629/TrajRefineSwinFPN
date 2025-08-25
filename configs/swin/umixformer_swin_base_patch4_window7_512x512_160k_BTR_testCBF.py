_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/BTR_testCBF512.py',
    '../_base_/default_runtime_swin.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False),
    decode_head=dict(
        type='APFormerHead2',  #FeedFormerHeadUNet, FeedFormerHeadUNetPlus, FeedFormerHead32, FeedFormerHead32_new'
        feature_strides=[4, 8, 16, 32],
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        decoder_params=dict(embed_dim=768, 
                            num_heads=[24, 12, 6, 3],
                            pool_ratio=[1, 2, 4, 8]),
        num_classes=2
        ),
    auxiliary_head=dict(in_channels=384, num_classes=2)
)

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
data = dict(samples_per_gpu=4, workers_per_gpu=1)
