# copied from uniformer
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/fpn_seg/configs/_base_/datasets/ade20k.py

# dataset settings
dataset_type = 'BTRDataset'
# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-20dB/'
# img_norm_cfg = dict(
#     mean=[62.8300745010376, 161.00881309509276, 191.23410959243773], std=[58.014750431853365, 41.70549256238703, 63.54621834926549], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-21dB/'
# img_norm_cfg = dict(
#     mean=[65.93851891342474, 164.0512605005381, 185.7472524059062], std=[61.47541852568047, 41.06393129699107, 65.78021825280449], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-22dB/'
# img_norm_cfg = dict(
#     mean=[68.17844266307597, 165.9393581468232, 182.0871586702308], std=[63.678619072078526, 40.627507405080095, 67.08925234836884], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-23dB/'
# img_norm_cfg = dict(
#     mean=[ 69.48995948324398, 167.07511629377092, 179.94204198097697], std=[66.22883910995213, 40.03996825325122, 68.51099108171181], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-24dB/'
# img_norm_cfg = dict(
#     mean=[ 70.58476179473254, 167.8108818677007, 178.3377342224121], std=[65.79030052087047, 40.102507534592924, 68.30057187363178], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-25dB/'
# img_norm_cfg = dict(
#     mean=[ 70.72562081473214, 167.89968177250452, 178.18152257374354], std=[65.92269401103894, 40.06786694723874, 68.37486576634114], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-26dB/'
# img_norm_cfg = dict(
#     mean=[ 71.07249253136771, 168.1461752482823, 177.62729508536202], std=[66.22883910995213, 40.03996825325122, 68.51099108171181], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-27dB/'
# img_norm_cfg = dict(
#     mean=[ 71.44335944192451, 168.40732199685615, 177.11960608499092], std=[66.44799978546001, 39.962803189896924, 68.64359865877726], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-28dB/'
# img_norm_cfg = dict(
#     mean=[ 71.3464510064376, 168.3251575001499, 177.27352905273438], std=[66.44799978546001, 39.962803189896924, 68.64359865877726], to_rgb=True)

data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/MVDR/-29dB/'
img_norm_cfg = dict(
    mean=[ 71.45173584787469, 168.4096988878752, 177.0826322321306], std=[66.50680427737423, 39.93952636411675, 68.68094530151726], to_rgb=True)


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
