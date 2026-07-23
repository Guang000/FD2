# Overall Directory Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PARENT_DIR")"
source "$ROOT_DIR/config.sh"
mkdir -p "$SCRIPT_DIR"/logs
DATASET_NAME=SC_imsize224
M_NUM=16
CAL_RATIO=4e-1
modelnames_lrs=("MobileNetV2 1e-3")
for modelname_lr in "${modelnames_lrs[@]}"; do
  read -r modelname lr <<< "${modelname_lr}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0 \
  python "$PARENT_DIR"/squeeze_cal.py \
      --dataset_name ${DATASET_NAME} \
      --dataset_dir ${Main_Data_Path}/${DATASET_NAME} \
      --save_dir ${Main_Data_Path}/pretrained_models/${DATASET_NAME} \
      --matplotlib \
      --model_list "${modelname}" \
      --model_source torchvision \
      --pretrained_weights \
      --pretrained_bn \
      --exp_name "${modelname}_M${M_NUM}_${CAL_RATIO}cal" \
      --M ${M_NUM} \
      --cal_ratio ${CAL_RATIO} \
      --master_port 29620 \
      --epoch 160 \
      --stop_epoch 50 \
      --batch_size 4 \
      --optimizer SGD \
      --world_size 1 \
      --lr "${lr}" > "$SCRIPT_DIR"/logs/squ_${DATASET_NAME}_"${modelname}"_M${M_NUM}_"${CAL_RATIO}"cal.log 2>&1
done
