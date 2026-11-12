
from datasets import load_from_disk, load_dataset
from kvpacket import KVPacket
from process_ds import ProcessDS
import sys
import os

if __name__=="__main__":

    
    if not os.path.exists("hotpot"):
        ProcessDS().make_ds({"hotpot_train": load_dataset("hotpotqa/hotpot_qa", "distractor", split="train"),
                    "hotpot_valid": load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")})
        
    ds = load_from_disk("hotpot").with_format("torch", device="cuda")
    
    todo = sys.argv[1]

    if todo == "kvpacket": 
        KVPacket().train(ds["hotpot_train"], ds["hotpot_valid"])
    

    

