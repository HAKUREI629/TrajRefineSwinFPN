# copied from uniformer
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/fpn_seg/configs/_base_/datasets/ade20k.py

# dataset settings
dataset_type = 'BTRDataset'
# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/simCBF/'
# img_norm_cfg = dict(
#     mean=[ 69.32788516, 165.80815822, 181.16055322], std=[64.25631563 ,41.08516009 ,67.71067907], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-20dB/'
# img_norm_cfg = dict(
#     mean=[  65.67843487, 162.15531639, 187.93148235], std=[60.85371001, 41.83973213, 65.62182621], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-21dB/'
# img_norm_cfg = dict(
#     mean=[  67.0305611038208, 163.65542736053467, 185.41638751983643], std=[62.33364621915915, 41.43552335616558, 66.52344981050722], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-22dB/'
# img_norm_cfg = dict(
#     mean=[  67.81597255706787, 164.3805139541626, 184.06926723480225], std=[63.151821824058835, 41.271994565898055, 67.00839631594711], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-23dB/'
# img_norm_cfg = dict(
#     mean=[  68.32293632507324, 164.73045471191406, 183.33490657806396], std=[63.58570858814837, 41.20463024232248, 67.29865710732835], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-24dB/'
# img_norm_cfg = dict(
#     mean=[  68.66977645874023, 165.09790855407715, 182.6907667541504], std=[63.95497203611568, 41.09646684470034, 67.4880049848232], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-25dB/'
# img_norm_cfg = dict(
#     mean=[  68.91504123687744, 165.32760753631592, 182.2872843170166], std=[64.17585107804011, 41.03578309236641, 67.61971600880783], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-26dB/'
# img_norm_cfg = dict(
#     mean=[  69.05580024719238, 165.46237445831298, 182.0432444000244], std=[64.33786071091383, 40.98062831855063, 67.70520448077963], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-27dB/'
# img_norm_cfg = dict(
#     mean=[  69.22609230041503, 165.53401252746582, 181.8340533065796], std=[64.43999757573233, 41.00185388412585, 67.78774818991342], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-28dB/'
# img_norm_cfg = dict(
#     mean=[  69.23742290496826, 165.68432518005372, 181.71216552734376], std=[64.53690460327026, 40.8753351906019, 67.8003797085764], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/CBF/-29dB/'
# img_norm_cfg = dict(
#     mean=[  69.16417381286621, 165.52927116394042, 181.88258296966552], std=[64.42722606083252, 40.99571904394311, 67.75298212093867], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Array/CBF/array1.5_32/'
# img_norm_cfg = dict(
#     mean=[  75.66657093048096, 168.80644165039064, 173.27528526306153], std=[69.6128142000712, 41.094708773531984, 70.65183383993173], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Array/CBF/array1.5_32_1.2_32/'
# img_norm_cfg = dict(
#     mean=[  69.03410106658936, 169.6334922027588, 177.83446483612062], std=[64.67343660601703, 38.212233948576106, 67.29451392455945], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Array/CBF/array2_64_circle/'
# img_norm_cfg = dict(
#     mean=[  103.22927433013916, 163.42913738250732, 155.57434017181396], std=[84.33773249188374, 55.35595871128088, 81.09901230986439], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Array/CBF/array0.2_32/'
# img_norm_cfg = dict(
#     mean=[  104.69572666168213, 160.2792529296875, 155.77558034261068], std=[84.62208967315776, 58.48903900906089, 81.70780240824297], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Array/CBF/array0.5_32/'
# img_norm_cfg = dict(
#     mean=[88.81312348683674, 164.47228956858316, 166.16665875752767], std=[77.60098642297474, 49.253714769387926, 76.52690035348309], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Array/CBF/array0.2_16/'
# img_norm_cfg = dict(
#     mean=[119.38187093098958, 158.2938540649414, 144.65514110565186], std=[88.98388096769744, 65.92743476467885, 84.76600365364945], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Multi/2/CBF/'
# img_norm_cfg = dict(
#     mean=[  69.77848424275716, 167.22032399495444, 179.6494567871094], std=[65.15826601990315, 40.24215228509917, 67.95378932392028], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Multi/3/CBF/'
# img_norm_cfg = dict(
#     mean=[  70.10834555589516, 167.46556035980925, 179.09931500267436], std=[65.38238412795661, 40.17318534542541, 68.08799769799994], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Multi/4/CBF/'
# img_norm_cfg = dict(
#     mean=[  67.99544982910156, 165.9913915846083, 182.15597678290473], std=[63.50884147522626, 40.55868980982143, 66.96998524327554], to_rgb=True)

# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Swellex/CBF/'
# img_norm_cfg = dict(
#     mean=[  67.93433148520333, 87.97745334534417, 220.69656101862589], std=[28.440968962785657, 43.849160700359974, 36.61223658727407], to_rgb=True)

data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Swellex/addnoise/'
img_norm_cfg = dict(
    mean=[  88.89753929138183, 101.01916801452637, 183.56205276489257], std=[91.2822480211234, 94.90366444049754, 85.69919544858458], to_rgb=True)


# data_root = '/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_global/CBF1024/'
# img_norm_cfg = dict(
#     mean=[ 50.92996697 ,163.22608654 ,201.12076145], std=[45.79858016 ,33.96943121, 54.28103141], to_rgb=True)


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
