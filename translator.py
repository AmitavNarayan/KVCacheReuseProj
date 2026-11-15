
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from datasets import Dataset, load_from_disk
import einx
import math
import sys

class Translator(torch.nn.Module):
    def __init__(self, num_layers: int, d_in: int, d_out: int, device: torch.device, dtype:torch.dtype):
        super().__init__()
        self.dtype = dtype
        mean = 0
        stdev = math.sqrt(2 / (d_in + d_out))
        self.weight = nn.Parameter(torch.empty((num_layers, d_out, d_in), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros((num_layers, d_out), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean, stdev, -3 * stdev, 3 * stdev)
    def forward(self, x: torch.Tensor):
        xw = einx.dot("l s d_in, l d_out d_in -> l s d_out", x, self.weight)
        y = xw + self.bias.unsqueeze(1) 
        return y

class Translate():
    def __init__(self, model_path: str = "Qwen2.5-1.5B-Instruct", 
                 alt_model_path: str = "Qwen2.5-Coder-1.5B-Instruct", model_dtype: torch.dtype = torch.bfloat16, compute_dtype: torch.dtype = torch.float64, device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=model_dtype).to(self.device)
        self.rope = sys.modules[type(self.model).__module__].apply_rotary_pos_emb
        self.alt_model = AutoModelForCausalLM.from_pretrained(alt_model_path, dtype=model_dtype).to(self.device)
        self.layers = self.model.config.num_hidden_layers 
        self.model.eval()
        self.alt_model.eval()
        self.device = self.model.device
        self.num_heads = self.model.config.num_key_value_heads
        self.d = self.num_heads * (self.model.config.hidden_size // self.model.config.num_attention_heads)
        self.w_k = None 
        self.w_v = None
        self.compute_dtype = compute_dtype
        # self.translator = Translator(self.layers, self.model.config.hidden_size, self.alt_model.config.hidden_size, self.device, dtype=torch.float32)
        # self.optim = torch.optim.AdamW(self.translator.parameters())
       
    def get_kv(self, example: dict):
        with torch.no_grad():
            caches = [m(example["input_ids"][None]).past_key_values.layers for m in (self.alt_model, self.model)]
            k, v = [[torch.stack([getattr(l, x) for l in c]).squeeze(1).to(dtype=torch.float64) for c in caches] for x in ["keys", "values"]]
            cos, sin = self.model.model.rotary_emb(k[0], torch.arange(k[0].shape[2])[None])
            k[0], k[1] = [self.rope(x, x, cos, -sin)[1] for x in [k[0], k[1]]]
            k[0], v[0] = [einx.id("l n t h, 1  -> l t ((n h) + 1)", x, torch.ones(1)) for x in [k[0], v[0]]]
        return k, v
    
    def train(self, train_dataset: Dataset, reg_param: float=1e-6):
        torch.set_default_dtype(self.compute_dtype)
        with torch.device(self.device):
            k, v =  [[torch.zeros(self.layers, self.d + 1, self.d + 1), torch.zeros(self.layers, self.num_heads, self.d + 1, self.d // self.num_heads)] for _ in range(2)]
            runsum = lambda x, ex: [x[0] + ex[0].transpose(-1,-2) @ ex[0], x[1] + ex[0].transpose(-1,-2).unsqueeze(1) @ ex[1]]
            for example in train_dataset:
                k_ex, v_ex = self.get_kv(example)
                k = runsum(k, k_ex)
                v = runsum(v, v_ex)
            
            reg_k, reg_v = [reg_param * x.diagonal(dim1=-2, dim2=-1) for x in [k[0],v[0]]]
            reg_k[:, -1] = 0
            reg_v[:, -1] = 0
            k[0], v[0] = k[0] + torch.diag_embed(reg_k), v[0] + torch.diag_embed(reg_v)
            self.w_k, self.w_v = [torch.linalg.solve(x[0].unsqueeze(1), x[1]) for x in [k,v]]
            torch.save(self.w_k, "/scratch/an4245/linear_translator_w_k.pt")
            torch.save(self.w_v, "/scratch/an4245/linear_translator_w_v.pt")

    def forward(self, k: torch.Tensor, v: torch.Tensor):
        k, v = k.to(self.compute_dtype), v.to(self.compute_dtype)
        k, v = [einx.id("l n t d, 1 -> l t ((n d) + 1)", x, torch.ones(1, device=k.device, dtype=self.compute_dtype)) for x in [k,v]]
        return [einx.dot("l t y, l h y e -> l h t e", x[0], x[1]).to(self.model.dtype) for x in [[k, self.w_k], [v, self.w_v]]]
        

if __name__=="__main__": 
    ds = load_from_disk("/scratch/an4245/webtext").with_format("torch", device="cuda")
    Translate().train(ds.select(range(400)))


   
                 