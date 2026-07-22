# change me if you want to run on different GPUS
Start_ipc=4
End_ipc=5
ipc=5
REC_NAME="rec_res18_MNV2_res50_DN121_ipc${ipc}"
# Overall Directory Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"
source "$SCRIPT_DIR"/constants.sh
syn_data_dir="${Main_Data_Path}/generated_data/syn_data/FADRMplus_FD2_${Dataset_Name}_09FC05_01SC4"
patch_dir=$Main_Data_Path/patches/$Dataset_Name
model_pool_dir=$Main_Data_Path/pretrained_models/$Dataset_Name
mkdir -p "$SCRIPT_DIR"/logs
# Remember to change the exp name
# Script Configuration
Log_NAME="FADRMplus_FD2_${REC_NAME}_ipcID_${Start_ipc}_${End_ipc}_09FC05_01SC4"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
CUDA_VISIBLE_DEVICES=0 \
python -u "$PARENT_DIR"/recover_FD2_FADRM.py \
    --matplotlib \
    --exp_name $REC_NAME \
    --apply_data_augmentation \
    --dataset_name "$Dataset_Name" \
    --class_num 100 \
    --subprocess_num 1 \
    --syn_data_path "$syn_data_dir" \
    --patch_dir "$patch_dir" \
    --model_pool_dir "$model_pool_dir" \
    --pretrained_model_type offline \
    --optimization_budgets 1500 1500 1500 2000 \
    --input_size_lis 200 224 200 224 \
    --alpha 0.5 \
    --model_choice ResNet18 MobileNetV2 ResNet50 Densenet121 \
    --M 8 16 8 16 \
    --cal_ratio 0.5 0.5 0.3 0.4 \
    --lr 0.1 \
    --r_bn 1e-3 \
    --FC \
    --FC_ratio 0.9 \
    --IntraFC_ratio 0.5 \
    --SC \
    --SC_ratio 0.1 \
    --SC_loss_threshold 0.0 \
    --store_best_images \
    --ipc_start $Start_ipc \
    --ipc_end $End_ipc \
    --initialisation_method "Patches" \
    --patch_diff "1" > "$SCRIPT_DIR"/logs/$Log_NAME.log 2>&1
