import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, DynamicCache
from datasets import Dataset, load_from_disk
import einx
import sys
import wandb
import copy

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
                 translator_types: list[str] = ["base", "no_sink"],
                 param_path: str = None):
        self.device = model.device
        self.dtype = model.dtype
        self.compute_dtype = compute_dtype
        self.d = model.config.num_key_value_heads * model.config.hidden_size // model.config.num_attention_heads
        self.rope = sys.modules[type(model).__module__].apply_rotary_pos_emb
        self.rotary_emb = model.model.rotary_emb
        self.model = model
        self.model.eval()
        self.translators = {}
        self.optims = {}
        for tr_type in translator_types:
            self.translators[tr_type] = nn.ModuleList([Translator(model.config.num_hidden_layers, self.d, self.device, dtype=self.compute_dtype) for _ in range(2)])
            self.optims[tr_type] = torch.optim.AdamW(self.translators[tr_type].parameters())
        if alt_model is not None: 
            self.alt_model = alt_model
            self.alt_model.eval()
        if param_path is not None:
            for tr_type in self.translators: 
                for i in range(2): self.translators[tr_type][i].load_state_dict(torch.load(f"{param_path}_{tr_type}.pt", map_location=self.device)[i])
        
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
    
    def train(self, train_dataset: Dataset, save_dir: str = "/scratch/an4245/translator_7B"):
        for example in train_dataset:
            tokens = example["input_ids"].to(self.device)
            kv = self.get_kv(tokens)
            for tr_type in self.translators:
                if tr_type=="base":
                    loss = sum([((self.translators[tr_type][i](kv[i][0]) - kv[i][1]) ** 2).sum() for i in range(2)]) / tokens.shape[0]
                if tr_type=="no_sink":
                    loss = sum([((self.translators[tr_type][i](kv[i][0])[:, :, 1:] - kv[i][1][:, :, 1:]) ** 2).sum() for i in range(2)]) / (tokens.shape[0] - 1)
                self.optims[tr_type].zero_grad()
                loss.backward()
                self.optims[tr_type].step()
        for tr_type in self.translators: torch.save([self.translators[tr_type][i].state_dict() for i in range(2)], f"{save_dir}_{tr_type}.pt")
        
        

    def translate(self, cache: DynamicCache, k_pos_ids: torch.Tensor, tr_type: str):
        kv = self.cache_to_tensor(cache)
        kv[0] = self.unrope(kv[0], k_pos_ids)
        for i in range(2): kv[i] = self.translators[tr_type][i](kv[i])
        kv[0] = self.unrope(kv[0], k_pos_ids, flip=True)
        kv = [x.to(self.dtype) for x in kv]
        for l, layer in enumerate(cache.layers): layer.keys, layer.values = kv[0][l][None], kv[1][l][None]
        return cache
        


