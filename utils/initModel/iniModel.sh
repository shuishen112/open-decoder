#!/bin/bash



# conda env set up
# source /sds_wangby/group_conda_envs/init.sh
# conda activate fapy310

experiment_name=Qwen2.5-1.5B-Instruct_dense
model_path=Qwen2.5-1.5B-Instruct-local
config_path=./src/model/phoenix11/final_config.json
# dir_name under src/model for modeling and config [phoenix11, phoenix11moe, phoenix12, phoenix12moe]
model_pattern=phoenix11
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