#sed -i 's/\r//' train.sh
export NOW=$(date '+%m-%d-%H:%M:%S')
export MODEL=swin.fpnasppv5
export OUTPUT_DIR=./outputs/seg/${MODEL}/02-26-11:39:37
export CKPT=${OUTPUT_DIR}/iter_32000.pth
export PYTHONPATH=/wangyunhao/cds/wangyunhao/code/BTR/pytorch-deeplab-xception-v1/Swin-Transformer-Semantic-Segmentation-swin/
CUDA_VISIBLE_DEVICES=1 python tools/train.py ./configs/swin/fpnasppv5_swin_base_patch4_window7_512x512_160k_BTR.py --work-dir $OUTPUT_DIR --gpu-ids 0 --resume $CKPT