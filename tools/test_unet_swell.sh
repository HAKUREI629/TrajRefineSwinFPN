#!/usr/bin/env bash
# sed -i 's/\r//' run.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export OUTPUT_DIR=./outputs/seg/unet.fcn/07-25-08:17:27
export TRUTH=/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Multi/2/CBF/
export CKPT=${OUTPUT_DIR}/iter_32000.pth
export NAME=SwellEX
export WORK_DIR=${OUTPUT_DIR}/testswellCBF32000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/unet/fcn_unet_s5-d16_512x512_40k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# # python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1 --threshold 1

export CKPT=${OUTPUT_DIR}/iter_64000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF64000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/unet/fcn_unet_s5-d16_512x512_40k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# # python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1 --threshold 1

export CKPT=${OUTPUT_DIR}/iter_96000.pth
export WORK_DIR=${OUTPUT_DIR}/testswellCBF96000/${NAME}/
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
rm -r ${WORK_DIR}
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/unet/fcn_unet_s5-d16_512x512_40k_BTR_testCBF.py $CKPT --show-dir $WORK_DIR
# python tools/calculatetrace.py --input $WORK_DIR --truth $TRUTH --single 0
python tools/calculatetrace.py --input $WORK_DIR --truth tools/truth_17_3_swell.mat --single 1 --threshold 1



# python tools/calculatetrace.py --input /wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/Swellex/OTA/ --truth tools/truth_17_3_swell.mat --single 1 --threshold 2