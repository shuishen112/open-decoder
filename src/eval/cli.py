import os
import platform
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import sys
import importlib
import argparse


def load_imodel_and_iconfig_package(model_pattern, src_path):
    # 动态构建模型路径
    model_path = os.path.join(src_path, 'model')
    
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
        IModelForCausalLM = importlib.import_module(f"{model_pattern}.modeling").IModelForCausalLM
        IConfig = importlib.import_module(f"{model_pattern}.configuration").IConfig
        # 返回导入的类
        return IModelForCausalLM, IConfig
    except ModuleNotFoundError as e:
        print(f"模块加载失败: {e}")
        return None, None
    
    
    
def load_model(model_name, device, num_gpus, model_pattern, src_path):
    if device == "cuda":
        kwargs = {"torch_dtype": torch.float32}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    elif device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {device}")

    IModelForCausalLM, _ = load_imodel_and_iconfig_package(model_pattern, src_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=True)
    model = IModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)

    if device == "cuda" and num_gpus == 1:
        model.cuda()

    return model, tokenizer


@torch.inference_mode()
def chat_stream(model, tokenizer, query, history, max_new_tokens=512,
                temperature=0.2, repetition_penalty=1.2, context_len=1024, stream_interval=2):
    
    prompt = generate_prompt(query, history, tokenizer.eos_token)
    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    device = model.device
    stop_str = tokenizer.eos_token
    stop_token_ids = [tokenizer.eos_token_id]

    l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[l_prompt:pos]
                    stopped = True
                else:
                    output = output[l_prompt:]
                yield output
            else:
                raise NotImplementedError

        if stopped:
            break

    del past_key_values


def generate_prompt(query, history, eos):
    if not history:
        return f"""<|im_start|>systemYou are a helpful assistant.<|im_end|><|im_start|>user: {query}<|im_end|><|im_start|>assistant: """
    else:
        prompt = '<|im_start|>systemYou are a helpful assistant.<|im_end|>'
        for i, (old_query, response) in enumerate(history):
            prompt += "<|im_start|>user: {} <|im_start|>assistant: {}".format(old_query, response) + eos
        prompt += "<|im_start|>user: {} <|im_start|>assistant: ".format(query)
        return prompt


def main(args):
    model, tokenizer = load_model(args.model_name, args.device, args.num_gpus, args.model_pattern, args.src_path)

    model = model.eval()

    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    history = []
    print("Hello, How can I help you? Enter 'clear' to clear the conversation history, 'stop' to terminate the program")
    while True:
        query = input("\n用户：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            os.system(clear_command)
            print("Model: Hello, How can I help you? Enter 'clear' to clear the conversation history, 'stop' to terminate the program")
            continue
        
        print(f"Model: ", end="", flush=True)
        pre = 0
        for outputs in chat_stream(model, tokenizer, query, history, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, repetition_penalty=1.2, context_len=1024):
            outputs = outputs.strip()
            # outputs = outputs.split("")
            now = len(outputs)
            if now - 1 > pre:
                print(outputs[pre:now - 1], end="", flush=True)
                pre = now - 1
        print(outputs[pre:], flush=True)
        history = history + [(query, outputs)]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/sds_wangby/models/Phoenix-II/ckpts/phoenix11-0.5B-3Loop-General")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--src_path", type=str, default="/sds_wangby/models/Phoenix-II/src")
    parser.add_argument("--model_pattern", type=str, default="phoenix11")
    parser.add_argument("--num-gpus", type=str, default="1")
    # parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    main(args)