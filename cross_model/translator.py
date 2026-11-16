import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
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
    def __init__(self, build_mode: bool = False,
                  model_path: str = "/scratch/an4245/Qwen2.5-3B-Instruct", 
                 alt_model_path: str = "/scratch/an4245/Qwen2.5-Coder-3B-Instruct", 
                 model_dtype: torch.dtype = torch.bfloat16, 
                 compute_dtype: torch.dtype = torch.float64, 
                 device: str = "cuda",
                 w_path: str = "/scratch/an4245/linear_translator_3B"):
        self.device = device
        self.dtype = model_dtype
        self.compute_dtype = compute_dtype
        config = AutoConfig.from_pretrained(model_path)
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(config)
        self.rope = sys.modules[type(m).__module__].apply_rotary_pos_emb
        self.rotary_emb = type(m.model.rotary_emb)(config).to(self.device)
        if build_mode:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=self.dtype).to(self.device)
            self.alt_model = AutoModelForCausalLM.from_pretrained(alt_model_path, dtype=self.dtype).to(self.device)
            self.layers = self.model.config.num_hidden_layers 
            self.model.eval()
            self.alt_model.eval()
            self.num_heads = self.model.config.num_key_value_heads
            self.d = self.num_heads * (self.model.config.hidden_size // self.model.config.num_attention_heads)
            # self.translator = Translator(self.layers, self.model.config.hidden_size, self.alt_model.config.hidden_size, self.device, dtype=torch.float32)
            # self.optim = torch.optim.AdamW(self.translator.parameters())
        else:
            self.w_k = torch.load(f"{w_path}_w_k.pt", map_location=self.device)
            self.w_v = torch.load(f"{w_path}_w_v.pt", map_location=self.device)

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
            k[0], v[0] = [einx.id("l n t h, 1  -> l t ((n h) + 1)", x, torch.ones(1, device=self.device)) for x in [k[0], v[0]]]
        return k, v
    
    def train(self, train_dataset: Dataset, reg_param: float=1e-6, save_dir: str = "/scratch/an4245/linear_translator_3B"):
        torch.set_default_dtype(self.compute_dtype)
        k, v =  [[torch.zeros((self.layers, self.d + 1, self.d + 1), device=self.device), torch.zeros((self.layers, self.num_heads, self.d + 1, self.d // self.num_heads), device=self.device)] for _ in range(2)]
        runsum = lambda x, ex: [x[0] + ex[0].transpose(-1,-2) @ ex[0], x[1] + ex[0].transpose(-1,-2).unsqueeze(1) @ ex[1]]
        for example in train_dataset:
            tokens = example["input_ids"].to(self.device)
            k_ex, v_ex = self.get_kv(tokens)
            k = runsum(k, k_ex)
            v = runsum(v, v_ex)
        
        reg_k, reg_v = [reg_param * x.diagonal(dim1=-2, dim2=-1) for x in [k[0],v[0]]]
        reg_k[:, -1] = 0
        reg_v[:, -1] = 0
        k[0], v[0] = k[0] + torch.diag_embed(reg_k), v[0] + torch.diag_embed(reg_v)
        self.w_k, self.w_v = [torch.linalg.solve(x[0].unsqueeze(1), x[1]) for x in [k,v]]
        torch.save(self.w_k, f"{save_dir}_w_k.pt")
        torch.save(self.w_v, f"{save_dir}_w_v.pt")

    def translate(self, kv: list[torch.Tensor], k_pos_ids: torch.Tensor):
        kv = [x.to(self.compute_dtype) for x in kv]
        kv[0] = self.unrope(kv[0], k_pos_ids)
        kv = [einx.id("l n t d, 1 -> l t ((n d) + 1)", x, torch.ones(1, device=x.device, dtype=self.compute_dtype)) for x in kv]
        kv = [einx.dot("l t y, l h y e -> l h t e", x, w) for x, w in [(kv[0], self.w_k), (kv[1], self.w_v)]]
        kv[0] = self.unrope(kv[0], k_pos_ids, flip=True)
        kv = [x.to(self.dtype) for x in kv]
        return kv
        

if __name__=="__main__":
    num_samples = 500
    ds = load_from_disk("/scratch/an4245/webtext").shuffle().select(range(num_samples)).with_format("torch", device="cuda")
    
    Translate(build_mode=True).train(ds)
    # Translate(build_mode=True).train(ds, save_dir="/scratch/an4245/linear_translator_0.5B_3B")