# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
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

import json
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers

# from fastchat.conversation import SeparatorStyle
# from fastchat.conversation import Conversation

IGNORE_TOKEN_ID = -100
DebugFlag = True

local_rank = None


def rank0_print(*args):
    # if local_rank == 0:
    #     print(*args)
    print(*args)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    global DebugFlag
    conv = Conversation(
        name="ichat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="<|im_end|>",
        sep2="<|endoftext|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
    )

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        try:
            source[0]["from"] != conv.roles[0]
        except Exception as e:
            print(
                f"\n***********************************************************************\n"
                f"source[0] Fail，Error Info: {str(e)}\n (Ignored) source:\n{source}\n"
                f"\n***********************************************************************\n"
            )
            source = [
                {"from": "human", "value": "who are you."},
                {
                    "from": "gpt",
                    "value": "I'm Phoenix-II. Nice to meet you, How can I help you?",
                },
            ]

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    inputs = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False,
    ).input_ids
    targets = inputs.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target, input in zip(conversations, targets, inputs):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = len(target) - total_len
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn, add_special_tokens=False).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded
            instruction_len = (
                len(tokenizer(parts[0], add_special_tokens=False).input_ids) - 2
            )

            if i != 0:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0:
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if DebugFlag:  # Inspect and check the correctness of masking
            inputStr = tokenizer.decode(input)
            targetStr = tokenizer.decode(target[target != IGNORE_TOKEN_ID])
            rank0_print(
                f"\n***********************************************************************\n"
                f"Debug:"
                f"\ninput:\n{input}"
                f"\ntarget:\n{target}"
                f"\ninputStr:\n{inputStr}"
                f"\ntargetStr:\n{targetStr}"
                f"\n***********************************************************************\n"
            )
            DebugFlag = False

    return dict(
        input_ids=inputs,
        labels=targets,
        attention_mask=inputs.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer) -> None:
        super(DataCollatorForSupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        batch_attn_mask = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(torch.bool)
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attn_mask,
            "labels": batch_labels,
        }
        return batch


block_size = 10


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def make_supervised_data_module_wiki(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:

    # load the wiki-texts datasets
    from datasets import load_dataset

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    # Add relevant_scores to tokenized datasets
    def add_relevant_scores(examples):
        examples["relevant_scores"] = examples["input_ids"].copy()
        return examples

    tokenized_datasets = tokenized_datasets.map(
        add_relevant_scores,
        batched=True,
        num_proc=4
    )
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1,
        num_proc=16,
    )


    # Split the dataset
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    return train_dataset, eval_dataset


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset if data_args.lazy_loading else SupervisedDataset
    rank0_print("Loading data...")

    if data_args.data_path.endswith("jsonl"):
        train_json = [json.loads(row) for row in open(data_args.data_path, "r")]
    else:
        train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.val_data_path:
        eval_json = json.load(open(data_args.val_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f"train_dataset: {len(train_dataset)}")
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
