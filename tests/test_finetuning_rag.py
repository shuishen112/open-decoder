from huggingface_hub.inference._generated.types import text_to_speech
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import Dataset
import torch


class S1Dataset(Dataset):
    def __init__(self, ds, tokenizer, max_length=4096):
        self.ds = ds
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sample = self.ds[index]
        question = sample["question"]
        answer = sample["answer"]
      
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )

        question = text.split("<|im_start|>assistant")[0]
        answer = "<|im_start|>assistant\n" + text.split("<|im_start|>assistant")[1]

        q_input_ids = self.tokenizer.encode(question)
        a_input_ids = self.tokenizer.encode(answer)

        input_ids = q_input_ids + a_input_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(q_input_ids) + a_input_ids


        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]
        else:
            padding_len = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            labels = labels + [-100] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.ds)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset("rag-datasets/rag-mini-wikipedia","question-answer")
    ds = ds['test']
    data_collator = DefaultDataCollator()

    args = TrainingArguments(
        output_dir="./s1",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        bf16=True,
        report_to="tensorboard",
    )

    train_dataset = S1Dataset(ds, tokenizer, max_length=252)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()

    #inference

    question = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What happened in 1833?"}],
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs.to(device), max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
