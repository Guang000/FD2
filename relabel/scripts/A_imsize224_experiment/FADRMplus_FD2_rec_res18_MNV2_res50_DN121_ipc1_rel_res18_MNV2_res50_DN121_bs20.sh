REC_NAME=rec_res18_MNV2_res50_DN121
REL_NAME=rel_res18_MNV2_res50_DN121
ipc=1
# Overall Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"
source "$SCRIPT_DIR"/constants.sh
bs=20
# Create logs directory
mkdir -p "$SCRIPT_DIR"/logs
Log_NAME=FADRMplus_FD2_${REC_NAME}_ipc${ipc}_${REL_NAME}_bs${bs}_09FC05_01SC4
WORLD_SIZE=1 \
RANK=0 \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
python "$PARENT_DIR"/relabel_FD2.py \
    --syn_data_path "${Generated_Path}/generated_data/syn_data/FADRMplus_FD2_${Dataset_Name}_09FC05_01SC4/${REC_NAME}_ipc${ipc}" \
    --fkd_path "${Generated_Path}/generated_data/new_labels/FADRMplus_FD2_${Dataset_Name}_09FC05_01SC4/${REC_NAME}_ipc${ipc}_${REL_NAME}" \
    --model_pool_dir "${Generated_Path}"/pretrained_models/"${Dataset_Name}" \
    --model_choice ResNet18 MobileNetV2 ResNet50 Densenet121 \
    --M 32 16 32 16 \
    --cal_ratio 0.3 0.2 0.4 0.4 \
    --workers 2 \
    --batch_size ${bs} \
    --dataset_name "${Dataset_Name}" \
    --start_epochs 0 \
    --end_epochs 400 \
    --fkd_seed 42 \
    --min_scale_crops 0.08 \
    --max_scale_crops 1 \
    --use_fp16 \
    --mode 'fkd_save' \
    --mix_type 'cutmix' > "${SCRIPT_DIR}"/logs/"${Log_NAME}".log 2>&1

