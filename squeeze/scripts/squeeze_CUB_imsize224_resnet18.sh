# Overall Directory Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PARENT_DIR")"
source "$ROOT_DIR/config.sh"
mkdir -p "$SCRIPT_DIR"/logs
DATASET_NAME=CUB_imsize224

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=3 \
python "$PARENT_DIR"/squeeze.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_dir ${Main_Data_Path}/${DATASET_NAME} \
    --save_dir ${Main_Data_Path}/pretrained_models/${DATASET_NAME} \
    --matplotlib \
    --model_list ResNet18 \
    --model_source torchvision \
    --pretrained_bn \
    --master_port 29620 \
    --epoch 100 \
    --batch_size 32 \
    --optimizer SGD \
    --world_size 1 \
    --lr 1e-2 > "$SCRIPT_DIR"/logs/${DATASET_NAME}_ResNet18.log 2>&1
