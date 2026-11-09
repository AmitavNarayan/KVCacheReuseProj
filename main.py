
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvlink import KVLink
from kvpacket import KVPacket
from process_ds import ProcessDS
import torch
import sys

def load_model(model_name: str, load_model: bool = True, load_tokenizer: bool = True, model_dtype=torch.bfloat16):
    retlist = []
    if load_model:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype)
            model = model.to("cuda")
            retlist.append(model)
    if load_tokenizer: 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        retlist.append(tokenizer)
    return retlist

if __name__=="__main__":
    instruction = "Answer the question using the documents."
    num_cpus = 12
    qwen_instruct = "Qwen/Qwen2.5-1.5B-Instruct"


    tokenizer = load_model(model_name=qwen_instruct, load_model=False)[0]
    datasets = {"hotpot_train": load_dataset("hotpotqa/hotpot_qa", "distractor", split="train"),
                "hotpot_valid": load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")}
    train_ds = ProcessDS(tokenizer, instruction).make_train_ds(num_cpus, datasets)
    
    method = sys.argv[1]
    if method == "norepair": 
        model, tokenizer = load_model(model_name=qwen_instruct, model_dtype=torch.float32)
        KVLink(model, tokenizer, num_link_tokens=0).train(train_ds["hotpot_train"], train_ds["hotpot_valid"])
    if method == "kvlink": 
        model, tokenizer = load_model(model_name=qwen_instruct, model_dtype=torch.float32)
        KVLink(model, tokenizer).train(train_ds["hotpot_train"], train_ds["hotpot_valid"])
    if method == "kvpacket": 
        model, tokenizer = load_model(model_name=qwen_instruct)
        KVPacket(model, tokenizer).train(train_ds["hotpot_train"], train_ds["hotpot_valid"])
    


    

