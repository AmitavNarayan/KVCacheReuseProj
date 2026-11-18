from helpers.translator import Translate
from reuse_methods.mymethod import MyMethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from helpers.process_ds import ProcessDS
if __name__=="__main__":
    num_samples, num_valid = 10000, 800
    load = lambda ds_name: load_from_disk(f"/scratch/an4245/{ds_name}").shuffle().select(range(num_samples)).with_format("torch", device="cuda").train_test_split(test_size=num_valid)
    model = AutoModelForCausalLM.from_pretrained("/scratch/an4245/Qwen2.5-7B-Instruct", device_map="cuda", dtype="bfloat16")
    alt_model = AutoModelForCausalLM.from_pretrained("/scratch/an4245/Qwen2.5-Coder-7B-Instruct", device_map="cuda", dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained("/scratch/an4245/Qwen2.5-7B-Instruct")
    
    webtext = load("webtext")
    translator = Translate(model=model, alt_model=alt_model)
    translator.train(webtext["train"], webtext["test"])

    musique = load("musique")
    MyMethod(model=model, alt_model=alt_model, tokenizer=tokenizer, translator=translator).run(musique["train"], musique["test"])