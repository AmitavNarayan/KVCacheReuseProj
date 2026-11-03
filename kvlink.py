

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from torch.utils.data import DataLoader
import einx

class KVLink():
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, num_link_tokens: int = 5, num_docs: int = 10, lr: float=5e-6):
        self.device = model.device
        self.model = model
        self.tokenizer = tokenizer
        self.num_link_tokens = num_link_tokens
        self.num_docs = num_docs    
        self.link_tokens = [f"<link{a+1}_{b+1}>" for a in range(num_docs) for b in range(num_link_tokens)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.link_tokens})
        self.model.resize_token_embeddings(len(tokenizer))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.link_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.link_tokens), device=self.device).reshape(num_docs, num_link_tokens)
        

    def get_attn_mask(self, prompt_len: int, prefix_len: int, doc_len: int, batch_size: int):
        block_len = doc_len + self.num_link_tokens
        row_mask = torch.arange(prompt_len, device=self.device).unsqueeze(1)
        col_mask = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        attn_mask = torch.tril(torch.ones((prompt_len, prompt_len), device=self.device, dtype=bool))
        col_mask = col_mask < (prefix_len + ((row_mask - prefix_len) // block_len) * block_len)
        row_mask = (row_mask >= prefix_len) & (row_mask < prefix_len + self.num_docs * block_len) & ((row_mask - prefix_len) % (block_len) < doc_len)
        attn_mask[col_mask & row_mask] = False
        attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(self.model.dtype)
        attn_mask = einx.id("... -> B 1 ...", attn_mask, B=batch_size)
        print(attn_mask.shape)
        return attn_mask

    def get_loss(self, answer_tokens: torch.Tensor, answer_logits: torch.Tensor):
        cel = torch.nn.CrossEntropyLoss(ignore_index = self.tokenizer.pad_token_id)
        logits = einx.id("B A V -> B V A", answer_logits.float())
        loss = cel(logits, answer_tokens)
        return loss
    
    def run(self, prefix_tokens: torch.Tensor, doc_tokens: torch.Tensor, suffix_tokens: torch.Tensor, answer_tokens: torch.Tensor):
        doc_len = doc_tokens.shape[2]
        doc_tokens = einx.id("A B C, B E -> A (B (C+E))", doc_tokens, self.link_ids)
        prompt_tokens = einx.id("A P, A D, A S -> A (P + D + S)", prefix_tokens, doc_tokens, suffix_tokens)

        attn_mask = self.get_attn_mask(prompt_tokens.shape[1], prefix_tokens.shape[1], doc_len, doc_tokens.shape[0])
        
        prefill = self.model(prompt_tokens, attention_mask=attn_mask, logits_to_keep=1)
        answer_logits = einx.id("B 1 V, B T V -> B (1+T) V", prefill.logits, self.model(answer_tokens, past_key_values=prefill.past_key_values).logits[:, :-1, :])
        loss = self.get_loss(answer_tokens, answer_logits)
        return loss
    
    def train(self, dataloader: DataLoader):
        wandb.init()
        self.model.train()
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(self.device)
            loss = self.run(batch["prefix"], batch["docs"], batch["suffix"], batch["answer"])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            wandb.log({"train-loss": loss.item()})
        wandb.finish()
        return self.model, self.optimizer
    







    