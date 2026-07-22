# change me if you want to run on different GPUS
Start_ipc=0
End_ipc=5
ipc=5
REC_NAME="rec_res18_ipc${ipc}"
# Overall Directory Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"
source "$SCRIPT_DIR"/constants.sh
syn_data_dir="$Main_Data_Path/generated_data/syn_data/SRe2Lplus_${Dataset_Name}"
patch_dir=$Main_Data_Path/patches/$Dataset_Name
model_pool_dir=${Main_Data_Path}/pretrained_models/${Dataset_Name}
mkdir -p "$SCRIPT_DIR"/logs
# Remember to change the exp name
# Script Configuration
Log_NAME="SRe2Lplus_${REC_NAME}_ipcID_${Start_ipc}_${End_ipc}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
CUDA_VISIBLE_DEVICES=3 \
python -u "$PARENT_DIR"/recover.py \
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
    --model_choice ResNet18 \
    --voter_type equal \
    --selected_size 1 \
    --lr 1e-3 \
    --iteration 10000 \
    --r_bn 1e-3 \
    --store_best_images \
    --ipc_start $Start_ipc \
    --ipc_end $End_ipc \
    --initialisation_method "Patches" \
    --patch_diff "2" > "$SCRIPT_DIR"/logs/$Log_NAME.log 2>&1
