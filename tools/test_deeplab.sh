#!/usr/bin/env bash
# sed -i 's/\r//' run.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export OUTPUT_DIR=./outputs/seg/deeplabv3plus.r50/01-20-11:37:22
export TRUTH=/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Multi/2/CBF/
# export CKPT=${OUTPUT_DIR}/iter_96000.pth
# export WORK_DIR=${OUTPUT_DIR}/testsimCBFnew/
# export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_BTR_testCBF512.py $CKPT --show-dir $WORK_DIR
export NAME=29dB
export CKPT=${OUTPUT_DIR}/iter_32000.pth
export WORK_DIR=${OUTPUT_DIR}/testsimCBF32000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_BTR_testCBF512.py $CKPT --show-dir $WORK_DIR
# python tools/calculatepd.py --npy_dir $WORK_DIR --mat_path tools/truth.mat --var_name truth -N 512 -M 1
# python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1 --threshold 2

export CKPT=${OUTPUT_DIR}/iter_64000.pth
export WORK_DIR=${OUTPUT_DIR}/testsimCBF64000/${NAME}/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_BTR_testCBF512.py $CKPT --show-dir $WORK_DIR
# python tools/calculatepd.py --npy_dir $WORK_DIR --mat_path tools/truth.mat --var_name truth -N 512 -M 1
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1 --threshold 2

export CKPT=${OUTPUT_DIR}/iter_96000.pth
export WORK_DIR=${OUTPUT_DIR}/testsimCBF96000/${NAME}/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_BTR_testCBF512.py $CKPT --show-dir $WORK_DIR
# python tools/calculatepd.py --npy_dir $WORK_DIR --mat_path tools/truth.mat --var_name truth -N 512 -M 1
#python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth.mat --single 1 --threshold 2