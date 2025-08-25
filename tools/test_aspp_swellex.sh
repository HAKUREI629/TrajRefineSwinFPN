#!/usr/bin/env bash
# sed -i 's/\r//' run.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export TRUTH=/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Multi/2/CBF/
export OUTPUT_DIR=./outputs/seg/swin.fpnasppv5/02-27-06:48:27
export NAME=SwellEX
export CKPT=${OUTPUT_DIR}/iter_16000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF16000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace_swell.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1 --threshold 1

export CKPT=${OUTPUT_DIR}/iter_32000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF32000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace_swell.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single True
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1 --threshold 1

export CKPT=${OUTPUT_DIR}/iter_48000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF48000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# #python tools/calculatetrace_swell.py --input $WORK_DIR --truth $TRUTH --single 0
# # python tools/calculatetrace_swell.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single True
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1 --threshold 1

export CKPT=${OUTPUT_DIR}/iter_64000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF64000/${NAME}/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# # python tools/calculatetrace_swell.py --input $WORK_DIR --truth $TRUTH --single 0
# # python tools/calculatetrace_swell.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single True
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1 --threshold 1

export CKPT=${OUTPUT_DIR}/iter_80000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF80000/${NAME}/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace_swell.py --input $WORK_DIR --truth $TRUTH --single 0
# python tools/calculatetrace_swell.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single True
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1 --threshold 1

export CKPT=${OUTPUT_DIR}/iter_96000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF96000/${NAME}/
# rm -r ${WORK_DIR}
# mkdir -p ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# # python tools/calculatetrace_swell.py --input $WORK_DIR --truth $TRUTH --single 0
# # python tools/calculatetrace_swell.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single True
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1 --threshold 1

# python tools/calculatetrace_swell.py --input  /wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Swellex/match/ --truth tools/truth_17_3_swell.mat --single 1
# python tools/calculatetrace_swell.py --input  /wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Swellex/OTA/ --truth tools/truth_17_3_swell.mat --single 1