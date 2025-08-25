#sed -i 's/\r//' train.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export OUTPUT_DIR=./outputs/seg
export MODEL=SUnetV3
export JOB_NAME=${MODEL}
export WORK_DIR=${OUTPUT_DIR}/${MODEL}/${NOW}
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
mkdir -p ${WORK_DIR}
CUDA_VISIBLE_DEVICES=1 python tools/train.py ./configs/SUnet/SUV3_base_patch4_window8_512x512_160k_BTR.py --work-dir $WORK_DIR --gpu-ids 0
