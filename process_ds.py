from transformers import PreTrainedTokenizer
from datasets import Dataset, concatenate_datasets
import torch


class ProcessDS:
    def __init__(self, tokenizer: PreTrainedTokenizer, inst: str):
        self.tokenizer = tokenizer
        self.inst = inst
    
    def tokenize(self, batch_text: list[str]):
        batch_tokens = []
        for text in batch_text:
            text_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            batch_tokens.append(text_tokens)
        return batch_tokens

    def make_docs(self, batch: dict[str, list], ds_name: str):
        batch_doctext = []
        if ds_name == "musique":
            for example_pars in batch['paragraphs']:
                batch_doctext.append(([par['title'] for par in example_pars], [par['paragraph_text'] for par in example_pars]))
        if ds_name == "hotpot":
            for example in batch['context']:
                batch_doctext.append(([title for title in example['title']], ["".join(sentences) for sentences in example['sentences']]))
        batch_docs = []
        for doc_title, doc_text in batch_doctext:
            docstrs = [title + "\n" + text for title, text in zip(doc_title, doc_text)]
            docs = [f"Document {i+1}: {doc}\n" for i, doc in enumerate(docstrs)]
            batch_docs.append(docs)
        return batch_docs
        
    def make_prompt(self, batch_questions: list[str], batch_docs: list[list[str]]):
        batch_prompts = []
        for example in zip(batch_questions, batch_docs):
            messages = [{"role": "system", "content": self.inst},
                                {"role": "user", "content": "".join(example[1])+f"Question: {example[0]}"}]
            batch_prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        return batch_prompts

    def get_boundary_tokens(self, batch_prompts: list[str]):
        batch_prefixes = []
        batch_suffixes = []
        for example in batch_prompts:
            prefix_idx = example.find("Document 1: ")
            suffix_idx = example.rfind("Question: ")
            batch_prefixes.append(example[:prefix_idx])
            batch_suffixes.append(example[suffix_idx:])
        batch_prefix_tokens = self.tokenize(batch_prefixes)
        batch_suffix_tokens = self.tokenize(batch_suffixes)
        return batch_prefix_tokens, batch_suffix_tokens

    def get_doc_tokens(self, batch_docs: list[list[str]]):
        batch_doc_tokens = []
        for example in batch_docs:
            batch_doc_tokens.append(self.tokenize(example))
            while len(batch_doc_tokens[-1]) < 10: batch_doc_tokens[-1].append([self.tokenizer.pad_token_id])
        return batch_doc_tokens

    def get_tokens_from_batch(self, batch: dict[str, list], ds_name: str) -> dict:
        batch_docs = self.make_docs(batch, ds_name)
        batch_prompts = self.make_prompt(batch['question'], batch_docs)
        batch_prefix_tokens, batch_suffix_tokens = self.get_boundary_tokens(batch_prompts)
        batch_doc_tokens = self.get_doc_tokens(batch_docs)
        batch_answer_tokens = self.tokenize(batch['answer'])
        return {"prefix": batch_prefix_tokens, "suffix": batch_suffix_tokens, "docs": batch_doc_tokens, "answer": batch_answer_tokens}

    def make_train_ds(self, num_cpus: int, ds: dict[str, Dataset]):
        train_ds = {}
        for ds_name in ds:
            batch_fn = lambda batch: self.get_tokens_from_batch(batch, ds_name)
            train_ds[ds_name] = ds[ds_name].map(batch_fn, num_proc=num_cpus, batched=True, remove_columns=ds[ds_name].column_names).with_format("torch")
        return train_ds
    
    


