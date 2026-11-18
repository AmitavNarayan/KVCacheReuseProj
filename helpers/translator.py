import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, DynamicCache
from datasets import Dataset, load_from_disk
import einx
import sys
import wandb

class Linear(torch.nn.Module):
    def __init__(self, num_layers: int, d: int,  device: torch.device, dtype:torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((num_layers, d, d), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros((num_layers, d), device=device, dtype=dtype))
    def forward(self, x: torch.Tensor):
        xw = einx.dot("l n_in t d_in, l (n_out d_out) (n_in d_in) -> l n_out t d_out", x, self.weight, n_out=x.shape[1], d_out=x.shape[3])
        y = einx.add("l n_out t d_out, l (n_out d_out) -> l n_out t d_out", xw, self.bias)
        return y

class Translator(torch.nn.Module):
    def __init__(self, num_layers: int, d: int,  device: torch.device, dtype:torch.dtype):
        super().__init__()
        self.first_linear = Linear(num_layers, d, device, dtype)
        self.second_linear = Linear(num_layers, d, device, dtype)
    def forward(self, x: torch.Tensor):
        y = x + self.second_linear(torch.relu(x + self.first_linear(x)))  
        return y

class Translate():
    def __init__(self, model: AutoModelForCausalLM,
                     alt_model: AutoModelForCausalLM = None, 
                 compute_dtype: torch.dtype = torch.float64, 
                 param_path: str = None):
        self.device = model.device
        self.dtype = model.dtype
        self.compute_dtype = compute_dtype
        self.d = model.config.num_key_value_heads * model.config.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers 
        self.rope = sys.modules[type(model).__module__].apply_rotary_pos_emb
        self.rotary_emb = model.model.rotary_emb
        self.model = model
        self.model.eval()
        self.translators = nn.ModuleList([Translator(self.layers, self.d, self.device, dtype=self.compute_dtype) for _ in range(2)])
        self.optim = torch.optim.AdamW(self.translators.parameters()) 
        self.alt_model = alt_model
        if alt_model is not None: 
            self.alt_model.eval()
        if param_path is not None:
            for i in range(2): self.translators[i].load_state_dict(torch.load(f"{param_path}.pt", map_location=self.device)[i])
        


    def unrope(self, k: torch.Tensor, pos_ids: torch.Tensor, flip: bool = False):
        cos, sin = self.rotary_emb(k, pos_ids[None])
        sign = 2 * int(flip) - 1
        k = self.rope(k, k, cos, sign * sin)[1]
        return k
    
    def cache_to_tensor(self, cache: DynamicCache):
        kv = [torch.stack([getattr(l, x) for l in cache.layers]).squeeze(1).to(self.compute_dtype) for x in ["keys", "values"]]
        return kv
        
    def get_kv(self, tokens: torch.Tensor):
        with torch.no_grad():
            caches = [m(tokens[None]).past_key_values for m in (self.alt_model, self.model)]
            k, v = map(list, zip(*[self.cache_to_tensor(c) for c in caches]))
            k_pos_ids = torch.arange(tokens.shape[0], device=self.device)
            k[0], k[1] = self.unrope(k[0], k_pos_ids), self.unrope(k[1], k_pos_ids)
        return k, v
    
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, valid_freq: float = 0.01, save_dir: str = "/scratch/an4245/translator_7B"):
        wandb.init()
        step = 1
        for example in train_dataset:
            tokens = example["input_ids"].to(self.device)
            kv = self.get_kv(tokens)
            loss = sum([((self.translators[i](kv[i][0]) - kv[i][1]) ** 2).sum() for i in range(2)])
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if step % int(valid_freq * len(train_dataset)) == 0:
                with torch.no_grad():
                    loss = [0,0]
                    num_tokens = 0
                    for example in valid_dataset:
                        tokens = example["input_ids"].to(self.device)
                        num_tokens += tokens.shape[0]
                        kv = self.get_kv(tokens)
                        for i in range(2): loss[i] += ((self.translators[i](kv[i][0]) - kv[i][1]) ** 2).sum()
                    for i in range(2): loss[i] = loss[i] / num_tokens
                    wandb.log({"k-loss": loss[0].item(), "v-loss": loss[1].item()})
            step += 1
            wandb.finish()
        torch.save([self.translators[i].state_dict() for i in range(2)], f"{save_dir}.pt")

    def translate(self, cache: DynamicCache, k_pos_ids: torch.Tensor):
        kv = self.cache_to_tensor(cache)
        kv[0] = self.unrope(kv[0], k_pos_ids)
        for i in range(2): kv[i] = self.translators[i](kv[i])
        kv[0] = self.unrope(kv[0], k_pos_ids, flip=True)
        kv = [x.to(self.dtype) for x in kv]
        for l, layer in enumerate(cache.layers): layer.keys, layer.values = kv[0][l][None], kv[1][l][None]
        return cache
        


