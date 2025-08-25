#!/usr/bin/env bash
# sed -i 's/\r//' run.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export OUTPUT_DIR=./outputs/seg/swin.fpn/02-17-03_24_05
export TRUTH=/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Multi/2/CBF/
# export CKPT=${OUTPUT_DIR}/iter_96000.pth
# export WORK_DIR=${OUTPUT_DIR}/testswellCBFnew/
# export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/swin/fpn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR

# export NOW=$(date '+%m-%d-%H:%M:%S')
# export OUTPUT_DIR=./outputs/seg/swin.fpnasppv4/01-24-03:00:43
export NAME=SwellEX
export CKPT=${OUTPUT_DIR}/iter_32000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF32000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fpn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1  --threshold 1

export CKPT=${OUTPUT_DIR}/iter_64000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF64000/${NAME}/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fpn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1  --threshold 1

export CKPT=${OUTPUT_DIR}/iter_96000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF96000/${NAME}/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fpn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1  --threshold 1