from transformers import PreTrainedTokenizer
from datasets import Dataset, concatenate_datasets
import torch


class ProcessDS:
    def __init__(self, tokenizer: PreTrainedTokenizer, inst: str):
        self.tokenizer = tokenizer
        self.inst = inst
    
    def tokenize(self, text: str):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def get_tokens(self, example: dict, ds_name: str) -> dict:
        doctext = ""
        if ds_name == "musique":
            doctext = [par['title'] for par in example['paragraphs']], [par['paragraph_text'] for par in example['paragraphs']]
        if ds_name == "hotpot":
            doctext = example['context']['title'], ["".join(sentences) for sentences in example['context']['sentences']]
        docs = [f"Document {i+1}: {title + "\n" + text}\n" for i, (title, text), in enumerate(zip(doctext[0], doctext[1]))]
        messages = [{"role": "system", "content": self.inst},
            {"role": "user", "content": "".join(docs)+f"Question: {example['question']}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_idx = prompt.find("Document 1: ")
        suffix_idx = prompt.rfind("Question: ")
        prefix_tokens = self.tokenize(prompt[:prefix_idx])
        suffix_tokens = self.tokenize(prompt[suffix_idx:])
        doc_tokens = self.tokenize(docs)
        answer_tokens = self.tokenize(example['answer'])+[self.tokenizer.eos_token_id]
        return {"prefix": prefix_tokens, "suffix": suffix_tokens, "docs": doc_tokens, "answer": answer_tokens}

    def make_train_ds(self, num_cpus: int, ds: dict[str, Dataset]):
        train_ds = {}
        for ds_name in ds:
            train_ds[ds_name] = ds[ds_name].map(self.get_tokens, fn_kwargs={"ds_name": ds_name}, num_proc=num_cpus, remove_columns=ds[ds_name].column_names).with_format("torch")
        return train_ds