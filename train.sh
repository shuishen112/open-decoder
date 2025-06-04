#!/bin/bash


experiment_name=qwen_dense # NameWay: PATTERN_BaseModel_LoopNum_Arch
WANDB_PROJECT=$experiment_name  # project name in wandb
MODEL_PATH=baseModels/doNotUpload/Qwen2.5-1.5B-Instruct_dense
DATA_PATH=./data/Train/General.json
MODEL_PATTERN=qwen_decoder # the path to the model in src/model
src_path=./src
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

# deepspeed ./src/train.py \
#     --deepspeed ./scripts/zero1.json \
python ./src/train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_pattern $MODEL_PATTERN \
    --src_path $src_path \
    --bf16 True \
    --output_dir ./ckpts/${experiment_name} \
    --run_name ${experiment_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_safetensors False \
    --gradient_checkpointing True \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --num_equal_loop_layers 84 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --model_max_length 4096 \
    --lazy_loading True \
    # --report_to wandb \
    
# > ${log_folder}/${log_name} 2>&1 & 

# Add --is_base True if training base model

## Other Useful Args
# --neftune_noise_alpha 10.0 \ # This can drastically improve model performance for instruction fine-tuning.
# --include_tokens_per_second True \ # Whether or not to compute the number of tokens per second per device for training speed metrics. 
# This will iterate over the entire training dataloader once beforehand, and will slow down the entire process.
# --torch_compile True \ # Whether or not to compile the model using PyTorch 2.0 torch.compile. This will use the best defaults for the torch.compile API.
# --gradient_checkpointing True \
# --max_grad_norm 1.0 \


# # MultiNode
# ssh check
# apt-get install pdsh
# chown root:root /usr/lib/x86_64-linux-gnu/pdsh
# chown root:root /usr/lib
# chmod 755 /usr/lib/x86_64-linux-gnu/pdsh
# chmod 755 /usr/lib

# deepspeed --hostfile hostfile \
#     ./src/train.py \
# ...