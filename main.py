
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from kvlink import KVLink
from kvpacket import KVPacket
from process_ds import ProcessDS
from collate_fn import collate
import torch

def load_model(model_name: str, device: str = "cuda", pad_token: str = "<|finetune_right_pad_id|>", load_model: bool = True, load_tokenizer: bool = True):
    retlist = []
    if load_model:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to(device)
            retlist.append(model)
    if load_tokenizer: 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = pad_token
        retlist.append(tokenizer)
    return retlist

if __name__=="__main__":
    instruction = "Answer the question using the documents."
    num_cpus = 12
    batch_size = 32
    qwen_instruct = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


    tokenizer = load_model(model_name=qwen_instruct, load_model=False)[0]
    datasets = {}
    datasets["musique"] = load_dataset("dgslibisey/MuSiQue", data_files="musique_ans_v1.0_train.jsonl", split="train")
    datasets["hotpot"] = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    train_ds = ProcessDS(tokenizer, instruction).make_train_ds(num_cpus, datasets)
    loader = {} 
    for key in datasets:
        loader[key] = DataLoader(dataset=train_ds[key], shuffle=True, batch_size=batch_size, collate_fn = lambda batch: collate(batch, tokenizer.pad_token_id))

    
    
    model, tokenizer = load_model(model_name=qwen_instruct)
    kvlink_model, kvlink_optim = KVLink(model, tokenizer).train(loader["hotpot"])
    torch.save(kvlink_model.state_dict(), "kvlink_model.pt")
    torch.save(kvlink_optim.state_dict(), "kvlink_optim.pt")
    model, tokenizer = load_model(model_name=qwen_instruct)
    kvpacket_model, kvpacket_optim = KVPacket(model, tokenizer).train(loader["musique"])
    torch.save(kvpacket_model.state_dict(), "kvpacket_model.pt")
    torch.save(kvpacket_optim.state_dict(), "kvpacket_optim.pt")


    

