# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# load_from = r"/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/ckpt/upernet_swin_base_patch4_window7_512x512.pth"
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
