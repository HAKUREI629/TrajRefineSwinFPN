#!/usr/bin/env bash
# sed -i 's/\r//' run.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export OUTPUT_DIR=./outputs/seg/SUnet/12-16-01:29:58
export CKPT=${OUTPUT_DIR}/iter_96000.pth
export WORK_DIR=${OUTPUT_DIR}/Swin_testimgCBF/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/SUnet/SU_base_patch4_window8_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR

