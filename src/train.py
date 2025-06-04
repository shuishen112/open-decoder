#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from dataclasses import dataclass, field
import pathlib
from typing import Optional
import torch
import transformers
import os
import sys
import wandb
wandb.init(mode="disabled")

from transformers import Trainer
from dataset.fs_dataset import (
    make_supervised_data_module,
    make_supervised_data_module_wiki,
)
import importlib


def load_imodel_and_iconfig_package(model_pattern, src_path):
    # 动态构建模型路径
    model_path = os.path.join(src_path, "model")

    # 判断路径是否存在
    if not os.path.exists(model_path):
        print(f"路径不存在: {model_path}")
        return None, None

    # 将该路径添加到 sys.path 中，确保 Python 可以找到这些模块
    if model_path not in sys.path:
        sys.path.append(model_path)

    # 动态导入模型和配置模块
    try:
        # 动态导入模型
        IModelForCausalLM = importlib.import_module(
            f"{model_pattern}.modeling"
        ).IModelForCausalLM
        IConfig = importlib.import_module(f"{model_pattern}.configuration").IConfig
        # 返回导入的类
        return IModelForCausalLM, IConfig
    except ModuleNotFoundError as e:
        print(f"模块加载失败: {e}")
        return None, None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-7B")
    enable_flash_attn: bool = field(default=False)
    is_base: bool = field(default=False)
    model_pattern: Optional[str] = field(default="phoenix")
    src_path: Optional[str] = field(default="phoenix")
    num_equal_loop_layers: Optional[int] = field(default=None)
    # loop_pattern: Optional[list] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    val_data_path: str = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    lazy_loading: bool = False
    system_prompt: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    checkpoint = None


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.is_base:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        IModelForCausalLM, IConfig = load_imodel_and_iconfig_package(
            model_args.model_pattern, model_args.src_path
        )
        config = IConfig.from_pretrained(model_args.model_name_or_path)
    enable_flash_attn = False
    if (
        model_args.enable_flash_attn
        and getattr(config, "_attn_implementation", None) is not None
    ):
        config._attn_implementation = "flash_attention_2"
        enable_flash_attn = True

    if model_args.is_base:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16 if enable_flash_attn else "auto",
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
    else:
        model = IModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16 if enable_flash_attn else "auto",
            cache_dir=training_args.cache_dir,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset, eval_dataset = make_supervised_data_module_wiki(
        tokenizer=tokenizer, data_args=data_args
    )


    # data_module = macke_pretrain_data_module(tokenizer, tokens_dataset=data_args.data_path, val_data_path=data_args.val_data_path)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
