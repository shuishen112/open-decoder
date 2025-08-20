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
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import Trainer
from dataset.fs_dataset import (
    make_supervised_data_module,
    make_supervised_data_module_wiki,
)

from dataset.demo_dataset import MultiHopDatasetWithSegments
import importlib


def load_imodel_and_iconfig_package(model_pattern, src_path):
    model_path = os.path.join(src_path, "model")

    if not os.path.exists(model_path):
        logger.error(f"path not found: {model_path}")
        return None, None

    if model_path not in sys.path:
        sys.path.append(model_path)

    try:
        IModelForCausalLM = importlib.import_module(
            f"{model_pattern}.modeling"
        ).IModelForCausalLM
        IConfig = importlib.import_module(f"{model_pattern}.configuration").IConfig

        return IModelForCausalLM, IConfig
    except ModuleNotFoundError as e:
        logger.error(f"module not found: {e}")
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


def inference():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments)
    )
    model_args, data_args = parser.parse_args_into_dataclasses()

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
            trust_remote_code=True,
        )
    else:
        model = IModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16 if enable_flash_attn else "auto",
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    context = """
        The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. 
        It was constructed in 1889 as the entrance arch to the 1889 World's Fair. 
        The tower is 330 meters tall, about the same height as an 81-story building. 
        Random fact: Elephants are large mammals. This information is not relevant.
        The Eiffel Tower was designed by Zhan Su, whose company also built the forbidden city.
        Another random fact: Pizza is a popular food. This is also not relevant to the question.
        The tower receives about 6 million visitors annually, making it one of the most visited monuments in the world.
    """
    question = "Who designed the Eiffel Tower and what else did his company build?"

    prompt = "Given the following context, answer the question: Context: " + context + "\n\nQuestion: " + question
    # prompt = question

    # prompt = "Who is Zhan Su?"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", return_offsets_mapping=True).to(model.device)

    # irrelevant tokens
    irrelevant_text = "The Eiffel Tower was designed by Zhan Su, whose company also built the forbidden city."
    start_char_idx = text.find(irrelevant_text) 
    end_char_idx = start_char_idx + len(irrelevant_text)

    input_ids = model_inputs['input_ids'][0]
    offset_mapping = model_inputs['offset_mapping'][0]

    start_token_idx = -1
    end_token_idx = -1

    # 遍历偏移量映射
    for i, (token_start_char, token_end_char) in enumerate(offset_mapping):
        # 检查当前 token 是否与目标文本段有重叠
        if token_start_char >= start_char_idx and token_end_char <= end_char_idx:
            # 如果是第一个找到的 token，记录起始索引
            if start_token_idx == -1:
                start_token_idx = i
            # 持续更新结束索引
            end_token_idx = i

    # 打印结果
    if start_token_idx != -1:
        print(f"原始文本段: '{text[start_char_idx:end_char_idx]}'")
        print(f"对应的 token 索q引范围是: [{start_token_idx}, {end_token_idx}]")
        print("对应的 tokens 是:", tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0][start_token_idx : end_token_idx + 1]))
    else:
        print("未找到对应的 tokens。")

    print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids))
    # print("Offsets:", offset_mapping)

    # add relevant scores
    relevant_scores = torch.ones(model_inputs.input_ids.shape[0], model_inputs.input_ids.shape[1], device=model.device)


    # set relevant scores to 0 for the last 10 tokens
    relevant_scores[:, start_token_idx:end_token_idx] = 0

    model_inputs["relevant_scores"] = relevant_scores.to(model.dtype)

    ## delete the offset_mapping
    del model_inputs['offset_mapping']

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=50
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == "__main__":
    inference()
