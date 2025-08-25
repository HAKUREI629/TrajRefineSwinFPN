_base_ = [
    '../_base_/datasets/BTR_testCBF512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
# norm_cfg = dict(type='BN', requires_grad=True)
# model = dict(
#     type='EncoderDecoder',
#     pretrained=None,
#     backbone=dict(
#         type='UNet',
#         in_channels=3,
#         base_channels=64,
#         num_stages=3,
#         strides=(1, 1, 1),
#         enc_num_convs=(2, 2, 2),
#         dec_num_convs=(2, 2),
#         downsamples=(True, True),
#         enc_dilations=(1, 1, 1),
#         dec_dilations=(1, 1),
#         with_cp=False,
#         conv_cfg=None,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         act_cfg=dict(type='ReLU'),
#         upsample_cfg=dict(type='InterpConv'),
#         norm_eval=False),
#     decode_head=dict(
#         type='FCNHead',
#         in_channels=64,
#         in_index=2,
#         channels=64,
#         num_convs=1,
#         concat_input=False,
#         dropout_ratio=0.1,
#         num_classes=2,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     train_cfg=dict(),
#     test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))

optimizer = dict(type='AdamW', lr=0.00006, weight_decay=0.01)

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=1, workers_per_gpu=1)