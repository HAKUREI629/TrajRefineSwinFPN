# copied from uniformer
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/fpn_seg/configs/_base_/datasets/ade20k.py

# dataset settings
dataset_type = 'BTRDataset'
# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-20dB/'
# img_norm_cfg = dict(
#     mean=[59.563187789916995, 148.2926070690155, 205.51793932914734], std=[50.101235155709475, 44.91131600523738, 57.93158663720939], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-21dB/'
# img_norm_cfg = dict(
#     mean=[61.24343171411631, 152.49475852810608, 200.34224303887814], std=[53.589727862756696, 44.40226086327894, 60.762410545429674], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-22dB/'
# img_norm_cfg = dict(
#     mean=[62.612697834871256, 155.03393033086036, 196.86502464450135], std=[55.877007080589884, 44.031102918921796, 62.47028294329239], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-23dB/'
# img_norm_cfg = dict(
#     mean=[63.49134460760622, 156.60235844826212, 194.6987196474659], std=[57.20981820191852, 43.7482407722658, 63.42798762082449], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-24dB/'
# img_norm_cfg = dict(
#     mean=[64.2174818077866, 157.7109883366799, 193.11121298342334], std=[58.28074863944557, 43.494137345406045, 64.15622945561685], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-25dB/'
# img_norm_cfg = dict(
#     mean=[ 64.32769775390625, 157.7814884185791, 192.93137768336706], std=[58.38138367187528, 43.523492875956364, 64.23593291701874], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-26dB/'
# img_norm_cfg = dict(
#     mean=[ 64.54435927527291, 158.1602235521589, 192.38561071668352], std=[58.71856934750018, 43.455426757555806, 64.43353063830723], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-27dB/'
# img_norm_cfg = dict(
#     mean=[64.89588774296276, 158.53992415311043, 191.79410184893692], std=[59.11980771392439, 43.35798558339477, 64.74103325985104], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-28dB/'
# img_norm_cfg = dict(
#     mean=[64.72615820901436, 158.38756240041633, 192.04793113574647], std=[58.93771699052502, 43.40389207879835, 64.5942606901634], to_rgb=True)

data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MUSIC/-29dB/'
img_norm_cfg = dict(
    mean=[64.83311763562654, 158.55941959849577, 191.79594856396056], std=[59.06026009512901, 43.363409987731785, 64.69639363419346], to_rgb=True)


crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='Resize', img_scale=(720, 720), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='AlignResize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))
