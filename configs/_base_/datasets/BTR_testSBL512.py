# copied from uniformer
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/fpn_seg/configs/_base_/datasets/ade20k.py

# dataset settings
dataset_type = 'BTRDataset'
# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-20dB/'
# img_norm_cfg = dict(
#     mean=[57.81463279724121, 132.32698950767517, 213.06672987937927], std=[40.03266002045128, 50.16917997705231, 49.4919661954731], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-21dB/'
# img_norm_cfg = dict(
#     mean=[57.99523139486507, 133.9608161303462, 211.63513814186564], std=[41.11885328120204, 50.30345343998698, 50.54866425852379], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-22dB/'
# img_norm_cfg = dict(
#     mean=[58.08115737292231, 134.96365286379444, 210.764054590342], std=[41.73316094686617, 50.35817198718259, 51.15500104009237], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-23dB/'
# img_norm_cfg = dict(
#     mean=[58.1699520033233, 135.49050529635682, 210.26409538424744], std=[42.08613622869249, 50.39736911327436, 51.507702334007746], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-24dB/'
# img_norm_cfg = dict(
#     mean=[58.22430131873306, 135.8559007450026, 209.93104389735632], std=[42.32753449814998, 50.410700807239536, 51.739359353563145], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-25dB/'
# img_norm_cfg = dict(
#     mean=[ 58.229001317705425, 135.99439477920532, 209.8214387212481], std=[42.41359409431015, 50.40815704077169, 51.81218248311564], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-26dB/'
# img_norm_cfg = dict(
#     mean=[ 58.25862673350743, 136.21760402406966, 209.63669054848808], std=[42.541459467130224, 50.40360101608915, 51.9452880305862], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-27dB/'
# img_norm_cfg = dict(
#     mean=[58.230834224767854, 136.2353142186215, 209.66023267779434], std=[42.5204904012931, 50.391750385324016, 51.91891397119303], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-28dB/'
# img_norm_cfg = dict(
#     mean=[58.1813441494055, 136.13190694440874, 209.7596638997396], std=[42.41367555587856, 50.394727849038475, 51.81972251226862], to_rgb=True)

data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-29dB/'
img_norm_cfg = dict(
    mean=[58.25645480239601, 136.20436946132727, 209.6646903188605], std=[42.547046175359334, 50.39203963288417, 51.941579630253585], to_rgb=True)

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
