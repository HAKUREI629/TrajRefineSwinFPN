# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='STUnet',
        pretrain_img_size=512,
        patch_size=4,
        embed_dim=128,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        res_num_layers=[3, 4, 6, 3],
        res_width_factor=0.5,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    decode_head=dict(
        type='STUnetHead',
        in_channels=[32, 32],
        in_index=[0, 1],
        input_transform='multiple_select',
        channels=16,
        decoder_channels=(512,256,128,64),
        skip_channels=[512,256,128,64],
        n_skip=4,
        zero_head=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
