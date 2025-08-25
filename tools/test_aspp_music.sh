#!/usr/bin/env bash
# sed -i 's/\r//' run.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export TRUTH=/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Multi/2/CBF/
export OUTPUT_DIR=./outputs/seg/swin.fpnasppv5/02-27-06:48:27
export CKPT=${OUTPUT_DIR}/iter_16000.pth
export NAME=29dB
export WORK_DIR=${OUTPUT_DIR}/diffDoA/testsimMUSIC16000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testMUSIC.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1

export CKPT=${OUTPUT_DIR}/iter_32000.pth
export WORK_DIR=${OUTPUT_DIR}/diffDoA/testsimMUSIC32000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testMUSIC.py $CKPT --show-dir $WORK_DIR
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1

export CKPT=${OUTPUT_DIR}/iter_48000.pth
export WORK_DIR=${OUTPUT_DIR}/diffDoA/testsimMUSIC48000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testMUSIC.py $CKPT --show-dir $WORK_DIR
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
# python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single True
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1

export CKPT=${OUTPUT_DIR}/iter_64000.pth
export WORK_DIR=${OUTPUT_DIR}/diffDoA/testsimMUSIC64000/${NAME}/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testMUSIC.py $CKPT --show-dir $WORK_DIR
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
# python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single True
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1

export CKPT=${OUTPUT_DIR}/iter_80000.pth
export WORK_DIR=${OUTPUT_DIR}/diffDoA/testsimMUSIC80000/${NAME}/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testMUSIC.py $CKPT --show-dir $WORK_DIR
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
# python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single True
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1

export CKPT=${OUTPUT_DIR}/iter_96000.pth
export WORK_DIR=${OUTPUT_DIR}/diffDoA/testsimMUSIC96000/${NAME}/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testMUSIC.py $CKPT --show-dir $WORK_DIR
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
# python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single True
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1