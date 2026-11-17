import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from datasets import Dataset, load_from_disk
import einx
import math
import sys

class Translator(torch.nn.Module):
    def __init__(self, num_layers: int, d: int,  device: torch.device, dtype:torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.weight = nn.Parameter(torch.zeros((num_layers, d, d), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros((num_layers, d), device=device, dtype=dtype))
    def forward(self, x: torch.Tensor):
        xw = einx.dot("l n_in t d_in, l (n_out d_out) (n_in d_in) -> l n_out t d_out", x, self.weight, n_out=x.shape[1], d_out=x.shape[3])
        y = einx.add("l n_out t d_out, l (n_out d_out) -> l n_out t d_out", xw, self.bias)
        return x+y

class Translate():
    def __init__(self, build_mode: bool = False,
                  model_path: str = "/scratch/an4245/Qwen2.5-7B-Instruct", 
                 alt_model_path: str = "/scratch/an4245/Qwen2.5-Coder-7B-Instruct", 
                 model_dtype: torch.dtype = torch.bfloat16, 
                 compute_dtype: torch.dtype = torch.float64, 
                 device: str = "cuda",
                 param_path: str = "/scratch/an4245/linear_translator_7B"):
        self.device = device
        self.dtype = model_dtype
        config = AutoConfig.from_pretrained(model_path)
        self.compute_dtype = compute_dtype
        self.num_heads = config.num_key_value_heads
        self.d = self.num_heads * config.hidden_size // config.num_attention_heads
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(config)
        self.rope = sys.modules[type(m).__module__].apply_rotary_pos_emb
        self.rotary_emb = type(m.model.rotary_emb)(config).to(self.device)
        self.layers = config.num_hidden_layers 
        self.translators = [Translator(self.layers, self.d, self.device, dtype=self.compute_dtype) for _ in range(2)]
        if build_mode:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=self.dtype).to(self.device)
            self.alt_model = AutoModelForCausalLM.from_pretrained(alt_model_path, dtype=self.dtype).to(self.device)
            self.model.eval()
            self.alt_model.eval()
            self.optims = [torch.optim.AdamW(self.translators[i].parameters()) for i in range(2)]
        else:
            for i in range(2): self.translators[i].load_state_dict(torch.load(f"{param_path}.pt", map_location=self.device)[i])


    def unrope(self, k: torch.Tensor, pos_ids: torch.Tensor, flip: bool = False):
        cos, sin = self.rotary_emb(k, pos_ids[None])
        sign = 2 * int(flip) - 1
        k = self.rope(k, k, cos, sign * sin)[1]
        return k
    
    def get_kv(self, tokens: torch.Tensor):
        with torch.no_grad():
            caches = [m(tokens[None]).past_key_values for m in (self.alt_model, self.model)]
            k, v = [[torch.stack([getattr(l, x) for l in c.layers]).squeeze(1).to(self.compute_dtype) for c in caches] for x in ["keys", "values"]]
            k_pos_ids = torch.arange(tokens.shape[0], device=self.device)
            k[0], k[1] = self.unrope(k[0], k_pos_ids), self.unrope(k[1], k_pos_ids)
        return k, v
    
    def train(self, train_dataset: Dataset, save_dir: str = "/scratch/an4245/linear_translator_7B"):
        torch.set_default_dtype(self.compute_dtype)
        for example in train_dataset:
            tokens = example["input_ids"].to(self.device)
            kv = self.get_kv(tokens)
            for i in range(2):
                loss = ((self.translators[i](kv[i][0]) - kv[i][1]) ** 2).mean()
                self.optims[i].zero_grad()
                loss.backward()
                self.optims[i].step()
        
        torch.save([self.translators[i].state_dict() for i in range(2)], f"{save_dir}.pt")

    def translate(self, kv: list[torch.Tensor], k_pos_ids: torch.Tensor):
        kv = [x.to(self.compute_dtype) for x in kv]
        kv[0] = self.unrope(kv[0], k_pos_ids)
        for i in range(2): kv[i] = self.translators[i](kv[i])
        kv[0] = self.unrope(kv[0], k_pos_ids, flip=True)
        kv = [x.to(self.dtype) for x in kv]
        return kv
        

if __name__=="__main__":
    num_samples = 10000
    ds = load_from_disk("/scratch/an4245/webtext").shuffle().select(range(num_samples)).with_format("torch", device="cuda")
    Translate(build_mode=True).train(ds, save_dir="/scratch/an4245/linear_translator_7B")