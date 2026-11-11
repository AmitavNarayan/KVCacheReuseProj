
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvlink import KVLink
from kvpacket import KVPacket
from process_ds import ProcessDS
from translator import Translate
import torch
import sys

def load_model(model_name: str, load_model: bool = True, load_tokenizer: bool = True, model_device="cuda", model_dtype=torch.bfloat16):
    retlist = []
    if load_model:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype)
            model = model.to(model_device)
            retlist.append(model)
    if load_tokenizer: 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        retlist.append(tokenizer)
    return retlist

if __name__=="__main__":
    instruction = "Answer the question using the documents."
    num_cpus = 12
    qwen_instruct = "Qwen/Qwen2.5-1.5B-Instruct"
    qwen_instruct_coder = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    tokenizer = load_model(model_name=qwen_instruct, load_model=False)[0]
    datasets = {"hotpot_train": load_dataset("hotpotqa/hotpot_qa", "distractor", split="train"),
                "hotpot_valid": load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")}
    ds = ProcessDS(tokenizer, instruction).make_ds(num_cpus, datasets)
    todo = sys.argv[1]

    if todo == "experiment":
        model, tokenizer = load_model(model_name = qwen_instruct)
        alt_model = load_model(model_name = qwen_instruct_coder, load_tokenizer=False)[0]
        Translate(model, alt_model).train(ds["hotpot_train"], ds["hotpot_valid"])
        
    if todo == "norepair": 
        model, tokenizer = load_model(model_name=qwen_instruct, model_dtype=torch.float32)
        alt_model = load_model(model_name = qwen_instruct_coder, load_tokenizer=False)[0]
        KVLink(model, alt_model, tokenizer, num_link_tokens=0).train(ds["hotpot_train"], ds["hotpot_valid"])

    if todo == "kvlink": 
        model, tokenizer = load_model(model_name=qwen_instruct, model_dtype=torch.float32)
        alt_model = load_model(model_name = qwen_instruct_coder, load_tokenizer=False)[0]
        KVLink(model, alt_model, tokenizer).train(ds["hotpot_train"], ds["hotpot_valid"])

    if todo == "kvpacket": 
        model, tokenizer = load_model(model_name=qwen_instruct)
        alt_model = load_model(model_name = qwen_instruct_coder, load_tokenizer=False)[0]
        KVPacket(model, alt_model, tokenizer).train(ds["hotpot_train"], ds["hotpot_valid"])
    

    

