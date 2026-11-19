from transformers import DynamicCache
import torch
from datasets import Dataset 
import torch.nn.functional as F 
from utils.template import Template


class KVLink(Template):
    def __init__(self, 
                 max_num_docs: int = 20,
                 num_link_tokens: int = 5, 
                 lr: float=5e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.variants = "main", "alt"
        self.model.gradient_checkpointing_enable()
        self.link_tokens = [f"<link{a+1}_{b+1}>" for a in range(max_num_docs) for b in range(num_link_tokens)]
        self.num_link_tokens = num_link_tokens
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.link_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.link_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.link_tokens), device=self.device, dtype=torch.long).reshape(max_num_docs, num_link_tokens)

        
    def build_cache(self, cache: DynamicCache, docs: list[torch.Tensor], variant: str):
        
        doc_ids = torch.tensor([], device=self.device, dtype=torch.long)
        ls_ids = torch.tensor([], device=self.device, dtype=torch.long)
        p = cache.get_seq_length()
        offset = p
        for doc in docs: 
            ls_ids = torch.cat((ls_ids, offset + torch.arange(doc.shape[0], doc.shape[0]+self.num_link_tokens, device=self.device)), dim=0)
            doc_ids = torch.cat((doc_ids, offset + torch.arange(doc.shape[0], device=self.device)), dim=0)
            offset += self.num_link_tokens + doc.shape[0]
        doc_mask = torch.block_diag(*[torch.ones((d.shape[0], d.shape[0]), device=self.device).tril() for d in docs]).bool()[None, None]
        write_model = self.alt_model if "alt" in variant else self.model
        doc_cache = write_model(input_ids=torch.cat(docs)[None], attention_mask=doc_mask, position_ids=doc_ids[None]).past_key_values
        for l, layer in enumerate(doc_cache.layers):
            cache.update(layer.keys, layer.values, l)
        ls = torch.cat([self.link_ids[i, :] for i in range(len(docs))])
        ls_mask = (torch.cat((torch.arange(p, device=self.device), doc_ids, ls_ids))[None] <= ls_ids[:, None])
        self.model(input_ids=ls[None], attention_mask = ls_mask[None, None], past_key_values=cache, position_ids=ls_ids[None], logits_to_keep=1)
        return cache

    
    def train(self, train_dataset: Dataset, step_size: int=32):
            for step, example in enumerate(train_dataset, start=1):
                answer_len = example["answer"].shape[0]
                segments = [example["prefix"], *[torch.cat((doc, self.link_ids[i, :])) for i, doc in enumerate(example["docs"])], torch.cat((example["suffix"], example["answer"]))]
                prompt = torch.cat(segments)
                mask = torch.block_diag(*[torch.ones((s.shape[0], s.shape[0]), device=self.device, dtype=torch.bool).tril() for s in segments])
                idx = example["prefix"].shape[0]
                for doc in example["docs"]: 
                    mask[idx+doc.shape[0]:idx+doc.shape[0]+self.num_link_tokens, :idx+doc.shape[0]]=1
                    idx += doc.shape[0] + self.num_link_tokens
                mask[-segments[-1].shape[0]:, :-segments[-1].shape[0]] = 1
                with torch.autocast(device_type=self.device, dtype=self.forward_dtype): 
                    output_logits = self.model(input_ids=prompt[None], attention_mask=mask[None, None], logits_to_keep=answer_len+1).logits[0, :-1]
                    loss = F.cross_entropy(output_logits, prompt[-answer_len:])
                (loss / step_size).backward()
                if step % step_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()


