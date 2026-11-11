
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvlink import KVLink
from kvpacket import KVPacket
from process_ds import ProcessDS
from translator import Translate
import pickle
import sys
import os

if __name__=="__main__":

    
    if not os.path.exists("datasets.pkl"):
        ProcessDS().make_ds({"hotpot_train": load_dataset("hotpotqa/hotpot_qa", "distractor", split="train"),
                    "hotpot_valid": load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")})
        
    ds = pickle.load("datasets.pkl")
    
    todo = sys.argv[1]
    
    if todo == "experiment":
        print("3")

    if todo == "kvpacket": 
        KVPacket().train(ds["hotpot_train"], ds["hotpot_valid"])
    

    

