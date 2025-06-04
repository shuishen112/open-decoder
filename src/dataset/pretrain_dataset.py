from typing import Dict
import torch, transformers
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from packed_dataset import TokenDatasets


IGNORE_INDEX = -100

def make_pretrain_data_module(tokenizer: PreTrainedTokenizer, tokens_dataset: str, val_data_path: str, max_seq_len: int) -> Dict:
    block_size, n_chunks = max_seq_len, 8
    train_dataset = TokenDatasets(tokens_dataset, n_chunks, block_size)
    data_collator = DataCollatorForPretrainDataset(tokenizer=tokenizer)
    if val_data_path is not None:
        val_dataset = ValidationDataset(torch.load(val_data_path), max_seq_len)
        return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)
    else:
        return dict(train_dataset=train_dataset, data_collator=data_collator)

class DataCollatorForPretrainDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer) -> None:
        super(DataCollatorForPretrainDataset, self).__init__()
        self.tokenizer = tokenizer

    def __call__(self, batch_input) -> Dict[str, torch.Tensor]:
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_attn_mask = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(torch.bool)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_input, batch_first=True, padding_value=IGNORE_INDEX)
        batch={
            "input_ids": batch_input_ids,
            "attention_mask": batch_attn_mask,
            "labels": batch_labels
        }
        return batch
    
class ValidationDataset(Dataset):
    def __init__(self, data_dict, max_len: int):
        super(ValidationDataset, self).__init__()
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.max_len = max_len

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        # return self.input_ids[i][:self.max_len]
        return dict(
            input_ids=self.input_ids[i][:self.max_len],
            labels=self.labels[i][:self.max_len],
            attention_mask=self.attention_mask[i][:self.max_len],
        )