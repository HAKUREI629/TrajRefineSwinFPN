_base_ = [
    '../_base_/models/fpn_swin_doublelabel.py', '../_base_/datasets/BTR_testCBF512.py',
    '../_base_/default_runtime_swin.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        patch_size=4,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(type='AdamW', lr=0.00006, weight_decay=0.01)

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=8, workers_per_gpu=2)
