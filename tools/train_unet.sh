#sed -i 's/\r//' train.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export OUTPUT_DIR=./outputs/seg
export MODEL=unet.fcn
export JOB_NAME=${MODEL}
export WORK_DIR=${OUTPUT_DIR}/${MODEL}/${NOW}
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/unet/fcn_unet_s5-d16_512x512_40k_BTR.py --work-dir $WORK_DIR --gpu-ids 0
