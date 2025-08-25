#!/usr/bin/env bash
# sed -i 's/\r//' run.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export OUTPUT_DIR=./outputs/seg/NGSwin/12-13-03:15:28
export CKPT=${OUTPUT_DIR}/iter_96000.pth
export WORK_DIR=${OUTPUT_DIR}/Swin_testimgCBF/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/NGSwin/ngswin_base_patch4_window8_ngram2_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR