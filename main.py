from cross_model_reuse.translator import Translate
from cross_model_reuse.mymethod import MyMethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

if __name__=="__main__":
     
    load = lambda ds_name, n, v: load_from_disk(f"/scratch/an4245/{ds_name}").shuffle().select(range(n)).with_format("torch", device="cuda").train_test_split(test_size=v)
    model = AutoModelForCausalLM.from_pretrained("/scratch/an4245/Qwen2.5-7B-Instruct", device_map="cuda", dtype="bfloat16")
    alt_model = AutoModelForCausalLM.from_pretrained("/scratch/an4245/Qwen2.5-Coder-7B-Instruct", device_map="cuda", dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained("/scratch/an4245/Qwen2.5-7B-Instruct")

    wbtx_n, wbtx_v = 5000,400
    webtext = load("webtext", wbtx_n, wbtx_v)
    translator = Translate(model=model, alt_model=alt_model)
    translator.train(webtext["train"])

    msq_n, msq_v = 801,800
    musique = load("musique", msq_n, msq_v) 
    MyMethod(model=model, alt_model=alt_model, tokenizer=tokenizer, translator=translator).run(musique["train"], musique["test"])