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
from dataset.dpo_dataset import DPODataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
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

class DPOTrainer:
    """Direct Preference Optimization Trainer"""
    
    def __init__(self, model, ref_model, tokenizer, beta=0.1):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta  # Temperature parameter
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def get_log_probs(self, model, input_ids, attention_mask, prompt_length):
        """Calculate log probabilities for the response tokens"""
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Get log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs for actual tokens
            gathered_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask out prompt tokens (we only care about response)
            response_mask = torch.zeros_like(shift_labels)
            for i in range(len(prompt_length)):
                response_mask[i, prompt_length[i]:] = 1
            
            # Apply attention mask
            response_mask = response_mask * attention_mask[..., 1:]
            
            # Sum log probs over response tokens
            response_log_probs = (gathered_log_probs * response_mask).sum(dim=-1)
            response_lengths = response_mask.sum(dim=-1)
            
            # Average log prob per token
            avg_log_probs = response_log_probs / (response_lengths + 1e-8)
            
            return avg_log_probs
    
    def dpo_loss(self, batch):
        """Calculate DPO loss"""
        
        # Get log probabilities from policy model
        chosen_log_probs = self.get_log_probs(
            self.model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['prompt_length']
        )
        
        rejected_log_probs = self.get_log_probs(
            self.model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['prompt_length']
        )
        
        # Get log probabilities from reference model
        chosen_ref_log_probs = self.get_log_probs(
            self.ref_model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['prompt_length']
        )
        
        rejected_ref_log_probs = self.get_log_probs(
            self.ref_model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['prompt_length']
        )
        
        # Calculate log ratios
        chosen_log_ratio = chosen_log_probs - chosen_ref_log_probs
        rejected_log_ratio = rejected_log_probs - rejected_ref_log_probs
        
        # DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()
        
        # Calculate accuracy (how often chosen is preferred)
        accuracy = (logits > 0).float().mean()
        
        return loss, accuracy, {
            'chosen_log_ratio': chosen_log_ratio.mean(),
            'rejected_log_ratio': rejected_log_ratio.mean(),
            'logits': logits.mean()
        }
    
    def train_step(self, batch, optimizer):
        """Single training step"""
        self.model.train()
        optimizer.zero_grad()
        
        loss, accuracy, metrics = self.dpo_loss(batch)
        loss.backward()
        optimizer.step()
        
        return loss.item(), accuracy.item(), metrics
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

        ref_model = transformers.AutoModelForCausalLM.from_pretrained(
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

        ref_model = IModelForCausalLM.from_pretrained(
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

    # please write the dpo training code here
    # Load preference dataset
    # Create dataset and dataloader
    example_data = [
        {"prompt": "Hello, how are you?", "chosen": "I'm good, thank you!", "rejected": "I'm not good, thank you!"},
        {"prompt": "What is your name?", "chosen": "My name is John.", "rejected": "My name is Jane."},
    ]
    dpo_dataset = DPODataset(example_data, tokenizer)
    dataloader = DataLoader(dpo_dataset, batch_size=2, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ref_model.to(device)

    # Initialize trainer
    trainer = DPOTrainer(model, ref_model, tokenizer, beta=0.1)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    
    # Training loop
    for epoch in range(20):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            device = next(model.parameters()).device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            loss, accuracy, metrics = trainer.train_step(batch, optimizer)
            
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}: "
                      f"Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        print(f"Epoch {epoch+1} completed: "
              f"Avg Loss = {avg_loss:.4f}, Avg Accuracy = {avg_accuracy:.4f}")
    
    return model

    
    



if __name__ == "__main__":
    train()
