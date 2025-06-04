#!/bin/bash


experiment_name=Qwen2.5-1.5B-Instruct_dense
model_path=Qwen2.5-1.5B-Instruct-local
config_path=./src/model/qwen_decoder/final_config.json
model_pattern=qwen_decoder
savedir_imodel=./baseModels/doNotUpload/${experiment_name}
src_path=./src
savepath_basemodel_namemodules=./utils/initModel/namemodules/basemodel.txt
savepath_imodel_namemodules=./utils/initModel/namemodules/imodel.txt

log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

python utils/initModel/iniModel.py \
    --savepath_basemodel_namemodules ${savepath_basemodel_namemodules} \
    --savepath_imodel_namemodules ${savepath_imodel_namemodules} \
    --model_name_or_path ${model_path} \
    --config_path ${config_path} \
    --model_pattern ${model_pattern} \
    --src_path ${src_path} \
    --savedir_imodel ${savedir_imodel} 
    
    # > ${log_folder}/${log_name} 2>&1 & 