_base_ = [
    '../_base_/models/fpn_cswin.py', '../_base_/datasets/BTR_testCBF512.py',
    '../_base_/default_runtime_swin.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        type='CSWin',
        embed_dim=96,
        depth=[2,4,32,2],
        num_heads=[4,8,16,32],
        split_size=[1,2,7,7],
        drop_path_rate=0.6,
        use_chk=False,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    decode_head=dict(num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data=dict(samples_per_gpu=2)

