from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from torch.utils.data import DataLoader

class KVLink():
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_num_docs: int=20, num_link_tokens: int = 5, lr: float=5e-6):
        self.device = model.device
        self.model = model
        self.model.gradient_checkpointing_enable()
        self.tokenizer = tokenizer
        self.num_link_tokens = num_link_tokens
        self.link_tokens = [f"<link{a+1}_{b+1}>" for a in range(max_num_docs) for b in range(num_link_tokens)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.link_tokens})
        self.model.resize_token_embeddings(len(tokenizer))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.link_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.link_tokens), device=self.device).reshape(max_num_docs, num_link_tokens)

    def get_attn_mask(self, pos: torch.Tensor):
        positions = pos
        row_mask = torch.arange(positions[0], positions[-3], device=self.device).unsqueeze(1)
        col_mask = torch.arange(0, positions[-3], device=self.device).unsqueeze(0)
        col_mask = col_mask < positions[torch.searchsorted(positions, row_mask, right=True)-1]
        row_mask = self.num_link_tokens < positions[torch.searchsorted(positions, row_mask, right=True)] - row_mask
        attn_mask = torch.tril(torch.ones((positions[-1], positions[-1]), device=self.device, dtype=torch.bool))
        attn_mask[positions[0]:positions[-3], :positions[-3]][col_mask & row_mask] = False
        return attn_mask

    def train(self, dataset: dict[str, torch.Tensor], step_size:int=32, compute_dtype=torch.bfloat16):
        wandb.init()
        self.model.train()
        loader = DataLoader(dataset, batch_size=None, shuffle=True)
        step = 1
        for example in loader:
            for key in example: 
                if key=="docs": 
                    for i, doc in enumerate(example["docs"]): example["docs"][i] = doc.to(self.device)
                else: example[key] = example[key].to(self.device)
            lens = [example["prefix"].shape[0], *[doc.shape[0]+self.num_link_tokens for doc in example["docs"]], example["suffix"].shape[0], example["answer"].shape[0]]
            example["docs"] = torch.cat([torch.cat((doc, self.link_ids[i, :])) for i, doc in enumerate(example["docs"])])
            prompt = torch.cat((example["prefix"], example["docs"], example["suffix"], example["answer"]))
            pos = torch.tensor(lens, device=self.device).cumsum(0)
            label = prompt.clone()
            label[:pos[-2]] = -100
            attn_mask = self.get_attn_mask(pos)
            with torch.autocast(device_type="cuda", dtype=compute_dtype): 
                loss = self.model(prompt[None, :], attention_mask=attn_mask[None, None, :, :], labels=label[None, :]).loss
            wandb.log({"train-loss": loss.item()})
            loss = loss / step_size
            loss.backward()
            if step % step_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            step += 1
        wandb.finish()
        return