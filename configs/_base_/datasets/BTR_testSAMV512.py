# copied from uniformer
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/fpn_seg/configs/_base_/datasets/ade20k.py

# dataset settings
dataset_type = 'BTRDataset'
# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-20dB/'
# img_norm_cfg = dict(
#     mean=[64.59303150177001, 86.0100658416748, 219.84018416404723], std=[22.29654762163921, 43.25635945038884, 33.58279662744849], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-21dB/'
# img_norm_cfg = dict(
#     mean=[64.38598453755282, 87.56357286414321, 220.1657330259985], std=[22.897649336745655, 43.888508068389974, 33.988586950222036], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-22dB/'
# img_norm_cfg = dict(
#     mean=[64.26549895928831, 88.28570035039162, 220.35247553611288], std=[23.136234080672747, 44.14073036287543, 34.12593399237967], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-23dB/'
# img_norm_cfg = dict(
#     mean=[64.19107701827069, 88.92317915935905, 220.41565782196668], std=[23.415094208474876, 44.42095249162037, 34.34085530731111], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-24dB/'
# img_norm_cfg = dict(
#     mean=[64.14528009842853, 89.19213477932676, 220.52921956899215], std=[23.466582362026696, 44.474171887565305, 34.36147567794604], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-25dB/'
# img_norm_cfg = dict(
#     mean=[ 64.12236608777728, 89.36985165732247, 220.539799622127], std=[23.56044187104741, 44.54928391238629, 34.429496452875014], to_rgb=True)

data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-26dB/'
img_norm_cfg = dict(
    mean=[64.09704810694645, 89.59861206590084, 220.61430593122515], std=[23.62029368595086, 44.6123096894368, 34.46932486540636], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-27dB/'
# img_norm_cfg = dict(
#     mean=[ 64.09945671898979, 89.48152187892369, 220.57527814592635], std=[23.57511064119892, 44.584448925352234, 34.433412404754705], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-28dB/'
# img_norm_cfg = dict(
#     mean=[64.0966986271373, 89.49434581555818, 220.613448226661], std=[23.547786274182116, 44.56339775959509, 34.416032020979124], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SAMV/-29dB/'
# img_norm_cfg = dict(
#     mean=[64.10644143087822, 89.55318906014426, 220.60766962954872], std=[23.62663532369717, 44.58911127547327, 34.469101162461286], to_rgb=True)

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
