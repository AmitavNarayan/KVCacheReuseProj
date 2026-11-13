
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


class ProcessDS:
    def __init__(self, inst: str = "Answer the question using the documents.", extra_inst: str = "\nRemember to keep answers short.",
                 tokenizer_path: str = "Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.inst = inst
        self.extra_inst = extra_inst
    
    def tokenize(self, text: str | list[str]):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def get_tokens(self, example: dict, ds_name: str) -> dict:
        doctext = ""
        if ds_name == "musique":
            doctext = [par['title'] for par in example['paragraphs']], [par['paragraph_text'] for par in example['paragraphs']]
        if ds_name == "hotpot_train" or ds_name == "hotpot_valid":
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

    def make_ds(self, 
                input_datasets: dict[str, Dataset],
                num_cpus: int = 12):
        datasets = {}
        for ds_name in input_datasets:
            datasets[ds_name] = input_datasets[ds_name].map(self.get_tokens, fn_kwargs={"ds_name": ds_name}, num_proc=num_cpus, remove_columns=input_datasets[ds_name].column_names).with_format("torch", device="cuda")
        DatasetDict(datasets).save_to_disk("hotpot")