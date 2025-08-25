#!/usr/bin/env bash
# sed -i 's/\r//' run.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export OUTPUT_DIR=./outputs/seg/swin.fpncat.doublelabel/01-15-13:34:39
export CKPT=${OUTPUT_DIR}/iter_128000.pth
export WORK_DIR=${OUTPUT_DIR}/testsimCBF/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/swin/fpncat_swin_base_patch4_window7_512x512_160k_BTRdoublelabel_testCBF.py $CKPT --show-dir $WORK_DIR
