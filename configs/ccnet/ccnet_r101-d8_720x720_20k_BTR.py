_base_ = './ccnet_r50-d8_720x720_20k_BTR.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
