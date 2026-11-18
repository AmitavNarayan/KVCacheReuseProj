
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

class ProcessDS:
    def __init__(self, tokenizer: AutoTokenizer,
                 inst: str = "Answer the question using the documents.", 
                 extra_inst: str = "\nRemember to keep answers short.",):
        self.tokenizer = tokenizer
        self.inst = inst
        self.extra_inst = extra_inst
    
    def tokenize(self, text: str | list[str]):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def get_tokens(self, example: dict, ds_name: str) -> dict:
        doctext = ""
        if ds_name == "musique":
            doctext = [par['title'] for par in example['paragraphs']], [par['paragraph_text'] for par in example['paragraphs']]
        if ds_name == "hotpot":
            doctext = example['context']['title'], ["".join(sentences) for sentences in example['context']['sentences']]
        docs = [f"Document {i+1}: {title + "\n" + text}\n" for i, (title, text), in enumerate(zip(doctext[0], doctext[1]))]
        messages = [{"role": "system", "content": self.inst+self.extra_inst},
            {"role": "user", "content": "".join(docs)+f"Question: {example['question']}"+self.extra_inst}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_idx = prompt.find("Document 1: ")
        suffix_idx = prompt.rfind("Question: ")
        prefix_tokens = self.tokenize(prompt[:prefix_idx])
        suffix_tokens = self.tokenize(prompt[suffix_idx:])
        doc_tokens = self.tokenize(docs)
        answer_tokens = self.tokenize(example['answer'])+[self.tokenizer.eos_token_id]
        return {"prefix": prefix_tokens, "suffix": suffix_tokens, "docs": doc_tokens, "answer": answer_tokens}

    def make_ds(self, dataset: Dataset,
                ds_name: str,
                save_path: str,
                num_cpus: int = 24):
        dataset.map(self.get_tokens, fn_kwargs={"ds_name": ds_name}, num_proc=num_cpus, remove_columns=dataset.column_names).save_to_disk(save_path)


if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("/scratch/an4245/Qwen2.5-7B-Instruct")
    musique = load_dataset("dgslibisey/MuSiQue", split="train+validation")
    hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train+validation")
    ProcessDS(tokenizer).make_ds(hotpot, "hotpot", "/scratch/an4245/hotpot")
    ProcessDS(tokenizer).make_ds(musique, "musique", "/scratch/an4245/musique")