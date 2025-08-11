"""
RAG-style generator-only training script for multi-hop QA (concatenated-context setup).

Expect dataset JSONL with one JSON object per line, each containing:
{
  "id": "...",
  "question": "...",
  "contexts": ["passage1 text", "passage2 text", ...],  # retrieved docs (frozen retriever output)
  "answer": "..."
}

Usage example:
python rag_gen_multihop_train.py \
  --train_file data/train.jsonl \
  --validation_file data/valid.jsonl \
  --model_name_or_path t5-base \
  --output_dir outputs/rag_gen \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --max_source_length 1024 \
  --max_target_length 64 \
  --num_train_epochs 3

Notes:
- This script concatenates retrieved passages with a separator token between them. If you prefer FiD (encode passages separately), ask and I can provide that variant.
- Retriever is assumed fixed; we only fine-tune the generator (encoder-decoder).
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
from torch.utils.data import Subset
import os

os.environ["WANDB_DISABLED"] = "0"

SEP = "</s>"  # separator token for concatenating passages (works for most tokenizers)


class MultiHopDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_source_length: int = 1024, max_target_length: int = 64, concat_fn=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.concat_fn = concat_fn if concat_fn is not None else self.default_concat

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                q = obj.get("question") or obj.get("query")
                contexts = obj.get("contexts") or obj.get("passages") or []
                answer = obj.get("answer") or obj.get("answers") or ""
                _id = obj.get("id")
                if q is None:
                    continue
                input_text = self.concat_fn(q, contexts)
                self.examples.append({"id": _id, "input_text": input_text, "answer": answer})

    def default_concat(self, question: str, contexts: List[str]) -> str:
        # Basic concat: Question first, then passages separated by SEP.
        # Format: "question \n <sep> passage1 <sep> passage2 ..."
        parts = [question.strip()]
        for p in contexts:
            if p is None:
                continue
            p = p.strip()
            if not p:
                continue
            parts.append(SEP)
            parts.append(p)
        return " \n ".join(parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        model_inputs = self.tokenizer(
            ex["input_text"],
            truncation=True,
            max_length=self.max_source_length,
            return_tensors=None,
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                ex["answer"],
                truncation=True,
                max_length=self.max_target_length,
                return_tensors=None,
            )
        model_inputs["labels"] = labels["input_ids"]
        # model_inputs["id"] = ex.get("id")
        return model_inputs


class MultiHopCausalDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length=1024, concat_fn=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concat_fn = concat_fn if concat_fn is not None else self.default_concat

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                q = obj.get("question") or obj.get("query")
                contexts = obj.get("contexts") or obj.get("passages") or []
                answer = obj.get("answer") or obj.get("answers") or ""
                _id = obj.get("id")
                if q is None:
                    continue
                # concat question, contexts, and answer
                input_text = self.concat_fn(q, contexts, answer)
                self.examples.append({"id": _id, "text": input_text})

    def default_concat(self, question, contexts, answer):
        # concat: Question + SEP + Contexts + SEP + Answer
        parts = [question.strip()]
        for p in contexts:
            if p:
                parts.append(SEP)
                parts.append(p.strip())
        parts.append(SEP)
        parts.append(answer.strip())
        return " \n ".join(parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]

        # generate labels, shift right by one, and pad with pad_token_id
        labels = input_ids[1:] + [self.tokenizer.pad_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

def get_qwen_data_collator(tokenizer):
    def qwen_data_collator(features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        batch_inputs = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        batch_labels = tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )

        batch = {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "labels": batch_labels["input_ids"],
        }
        return batch
    return qwen_data_collator
def normalize_answer(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(s.split())
    return s


def compute_em_f1(preds: List[str], golds: List[str]):
    # returns micro EM and F1 (simple token-level F1)
    em = 0
    f1s = []
    for p, g in zip(preds, golds):
        np_ = normalize_answer(p)
        ng = normalize_answer(g)
        if np_ == ng:
            em += 1
            f1s.append(1.0)
            continue
        p_tokens = np_.split()
        g_tokens = ng.split()
        common = set(p_tokens) & set(g_tokens)
        num_same = sum(min(p_tokens.count(w), g_tokens.count(w)) for w in common)
        if num_same == 0:
            f1s.append(0.0)
        else:
            precision = num_same / max(1, len(p_tokens))
            recall = num_same / max(1, len(g_tokens))
            f1 = 2 * precision * recall / (precision + recall)
            f1s.append(f1)
    em_score = em / max(1, len(preds))
    f1_score = float(np.mean(f1s))
    return {"em": em_score, "f1": f1_score}


def postprocess_text(preds, labels, tokenizer):
    # print(preds)
    preds[preds == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # strip
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [lab.strip() for lab in decoded_labels]
    return decoded_preds, decoded_labels


def make_argparser():
    parser = argparse.ArgumentParser(description="Train generator-only RAG for multi-hop QA")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output_dir", type=str, default="outputs/rag_gen")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=500)
    return parser


def main():
    parser = make_argparser()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to("cuda")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to("cuda")

    train_dataset = MultiHopCausalDataset(args.train_file, tokenizer, max_length=args.max_source_length+args.max_target_length)
    eval_dataset = MultiHopCausalDataset(args.validation_file, tokenizer, max_length=args.max_source_length+args.max_target_length)
 

    # Suppose `my_dataset` is your Dataset instance
    subset_indices = list(range(100))  # first 1000 samples
    eval_dataset = Subset(eval_dataset, subset_indices)
    # Data collator will handle padding; ensures labels are shifted in the model
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        logging_steps=args.logging_steps,
        save_steps=500,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        seed=args.seed,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"  # wandb
    )

    # simple compute_metrics using EM / F1
    metric = evaluate.load("rouge")  # we use rouge only for logging; EM/F1 below

    def compute_metrics(eval_preds):
        # eval_preds: (predictions, labels, metrics?) depending on Trainer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds, decoded_labels = postprocess_text(preds, labels, tokenizer)
        scores = compute_em_f1(decoded_preds, decoded_labels)
        # also include rougeL
        rouge_scores = metric.compute(predictions=decoded_preds, references=decoded_labels)
        # rouge gives a dict; pick rougeL
        scores.update({"rougeL": rouge_scores["rougeL"]})
        return scores


    data_collator = get_qwen_data_collator(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()

    # Quick evaluation on validation set with generate
    print("Running final evaluation...")
    preds = trainer.predict(eval_dataset)
    pred_texts, label_texts = postprocess_text(preds.predictions, preds.label_ids, tokenizer)
    scores = compute_em_f1(pred_texts, label_texts)
    print("Validation EM / F1:", scores)


if __name__ == "__main__":
    main()
