
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvlink import KVLink
from kvpacket import KVPacket
from process_ds import ProcessDS
import torch

def load_model(model_name: str, load_model: bool = True, load_tokenizer: bool = True, model_dtype=torch.bfloat16):
    retlist = []
    if load_model:
            model = AutoModelForCausalLM.from_pretrained(model_name, model_dtype)
            model = model.to("cuda")
            retlist.append(model)
    if load_tokenizer: 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        retlist.append(tokenizer)
    return retlist

if __name__=="__main__":
    instruction = "Answer the question using the documents."
    num_cpus = 12
    qwen_instruct = "Qwen/Qwen2.5-1.5B-Instruct"


    tokenizer = load_model(model_name=qwen_instruct, load_model=False)[0]
    datasets = {}
    datasets["musique"] = load_dataset("dgslibisey/MuSiQue", data_files="musique_ans_v1.0_train.jsonl", split="train")
    datasets["hotpot"] = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    train_ds = ProcessDS(tokenizer, instruction).make_train_ds(num_cpus, datasets)
    
    model, tokenizer = load_model(model_name=qwen_instruct, model_dtype=torch.float32)
    KVLink(model, tokenizer).train(train_ds["hotpot"])
    


    

