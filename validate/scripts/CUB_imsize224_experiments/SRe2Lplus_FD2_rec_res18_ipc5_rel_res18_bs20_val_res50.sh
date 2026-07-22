# Overall Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"
source "$SCRIPT_DIR"/constants.sh
ipc=5
REC_NAME=rec_res18
bs=20
REL_NAME=rel_res18
VAL_NAME=val_ResNet50
EXP_NAME=SRe2Lplus_FD2_${REC_NAME}_ipc${ipc}_${REL_NAME}_bs${bs}_${VAL_NAME}_09FC05_01SC4
Model_Name=ResNet50
ODP=${Generated_Data_Path}/syn_data/SRe2Lplus_FD2_${Dataset_Name}_09FC05_01SC4/${REC_NAME}_ipc${ipc}
FKD=${Generated_Data_Path}/new_labels/SRe2Lplus_FD2_${Dataset_Name}_09FC05_01SC4/${REC_NAME}_ipc${ipc}_${REL_NAME}_bs${bs}_ipc${ipc}
OPD=${Generated_Data_Path}/validate_output
mkdir -p "$SCRIPT_DIR"/logs
PROJECT="SRe2Lplus_FD2_${Dataset_Name}_${VAL_NAME}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
CUDA_VISIBLE_DEVICES=0 \
python "$PARENT_DIR"/train_fkd_FD2.py \
    --model $Model_Name \
    --model_source torchvision \
    --fkd_source backbone \
    --ipc $ipc \
    --matplotlib \
    --project "$PROJECT" \
    --exp_name "$EXP_NAME" \
    --original_data_path "$ODP" \
    --fkd_path "$FKD" \
    --output_dir "$OPD" \
    --batch_size "$bs" \
    --epochs 400 \
    --dataset_name "$Dataset_Name" \
    --gradient_accumulation_steps 2 \
    --mix_type 'cutmix' \
    --cos \
    --eta 2.0 \
    --workers 2 \
    --temperature 20 \
    --lr 1e-3 \
    --momentum 0.9 \
    --weight_decay 1e-5 \
    --val_dir "$val_dir" > "$SCRIPT_DIR"/logs/"$EXP_NAME".log 2>&1

