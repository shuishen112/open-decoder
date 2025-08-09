import json
import argparse
from datasets import load_dataset


def para_from_context(context):
    """Hotpot context format: list of [title, [sent1, sent2, ...]] pairs."""
    paras = []
    for title, sents in context:
        text = " ".join(sents).strip()
        if text:
            paras.append({"title": title, "text": text})
    return paras
def gold_paragraphs_from_example(ex):
    # ex has fields: 'supporting_facts' list of [title, sent_idx]
    # and 'context' which is a list of [title, [sents...]]
    ctx_paras = para_from_context(ex.get("context", []))
    # map title -> paragraph text
    title2para = {p["title"]: p["text"] for p in ctx_paras}
    gold_paras = []
    for title, sid in ex.get("supporting_facts", []):
        if title in title2para:
            gold_paras.append(title2para[title])
    # dedupe
    seen = set()
    final = []
    for p in gold_paras:
        if p not in seen:
            final.append(p)
            seen.add(p)
    return final

def convert_hotpotqa_to_jsonl(dataset: str, output_file: str, max_examples: int = None):
    # Load HotpotQA from Hugging Face Datasets
    
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            question = example["question"].strip()
            answer = example["answer"].strip()
            # HotpotQA provides supporting_facts as titles + sentence ids
            # We'll extract gold paragraphs as contexts
            contexts = []
            tmp = []
            contexts = {name: "".join(sents) for name, sents in example["context"]}
            for fact_name, _sent_id in example["supporting_facts"]:
                psg = contexts[fact_name]
                tmp.append(psg)
            golden_passages = []
            for p in tmp:
                if p not in golden_passages:
                    golden_passages.append(p)
            # For simplicity, we keep all gold paragraphs (many will be irrelevant for multi-hop)
            obj = {
                "id": example["_id"],
                "question": question,
                "contexts": golden_passages,
                "answer": answer
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            if max_examples and count >= max_examples:
                break
    print(f"Saved {count} examples to {output_file}")


def load_hotpot(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HotpotQA to RAG generator JSONL format")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_file", type=str, default="hotpotqa_train.jsonl")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--input_file", type=str, default="data/hotpotqa/hotpot_train_v1.1.json")
    args = parser.parse_args()
    # convert_hotpotqa_to_jsonl(args.split, args.output_file, args.max_examples)
    data = load_hotpot(args.input_file)
    convert_hotpotqa_to_jsonl(data, args.output_file, args.max_examples)