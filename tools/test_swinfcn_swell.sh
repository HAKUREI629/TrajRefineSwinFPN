#!/usr/bin/env bash
# sed -i 's/\r//' run.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export OUTPUT_DIR=./outputs/seg/swin.fcn/08-08-02:10:57
export TRUTH=/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Multi/2/CBF/
export CKPT=${OUTPUT_DIR}/iter_16000.pth
export NAME=SwellEX
export WORK_DIR=${OUTPUT_DIR}/testsimCBF16000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/swin/fcn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1 --threshold 2

export NOW=$(date '+%m-%d-%H:%M:%S')
# export OUTPUT_DIR=./outputs/seg/swin.upernet/01-03-01_06_13
export CKPT=${OUTPUT_DIR}/iter_32000.pth
export WORK_DIR=${OUTPUT_DIR}/testsimCBF32000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fcn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1 --threshold 2

export CKPT=${OUTPUT_DIR}/iter_48000.pth
export WORK_DIR=${OUTPUT_DIR}/testsimCBF48000/${NAME}/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fcn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1 --threshold 2

export CKPT=${OUTPUT_DIR}/iter_64000.pth
export WORK_DIR=${OUTPUT_DIR}/testsimCBF64000/${NAME}/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fcn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1 --threshold 2

export CKPT=${OUTPUT_DIR}/iter_80000.pth
export WORK_DIR=${OUTPUT_DIR}/testsimCBF80000/${NAME}/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fcn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1 --threshold 2

export CKPT=${OUTPUT_DIR}/iter_96000.pth
export WORK_DIR=${OUTPUT_DIR}/testsimCBF96000/${NAME}/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fcn_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1 --threshold 2