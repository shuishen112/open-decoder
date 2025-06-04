import torch, os, math, io, json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import importlib
MODEL_PATTERN=phoenix12 # directory name under src/model [phoenix11, phoenix11moe, phoenix12, phoenix12moe]

global local_rank
def rank0_print(*args):
    if local_rank == 0:
        print(*args)
    
    
def print_trainable_parameters(model, desc:str=""):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    desc = f"[{desc}] " if len(desc) > 0 else ""
    print(
        f"{desc}trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return trainable_params, all_param
        
        
        
def load_model(args):
    model_path = args.model_path
    
    if args.checkpoint_path:
        assert os.path.isdir(os.path.join(args.checkpoint_path, "tfmr")), f"Model weights not found at: {args.checkpoint_path}"
        model_path = os.path.join(args.checkpoint_path, "tfmr")
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = not args.gradient_checkpointing
    
    enable_flash_attn = False
    if args.enable_flash_attn and getattr(config, '_attn_implementation', None) is not None:
        config._attn_implementation = "flash_attention_2"
        enable_flash_attn = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype=torch.bfloat16 if enable_flash_attn else 'auto',
        cache_dir=".cache", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=args.max_seq_len, padding_side="right", use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict