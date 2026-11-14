import wandb
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from datasets import Dataset, load_from_disk
import einx
import math

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
        self.alt_model = AutoModelForCausalLM.from_pretrained(alt_model_path, dtype=model_dtype).to(self.device)
        self.layers = self.model.config.num_hidden_layers + 1
        self.model.eval()
        self.alt_model.eval()
        self.device = self.model.device
        self.d = self.model.config.hidden_size
        self.compute_dtype = compute_dtype
        # self.translator = Translator(self.layers, self.model.config.hidden_size, self.alt_model.config.hidden_size, self.device, dtype=torch.float32)
        # self.optim = torch.optim.AdamW(self.translator.parameters())
       

    def get_x_y(self, example: dict):
        with torch.no_grad():
            tokens = example["input_ids"][None]
            x = torch.stack(self.model(tokens, output_hidden_states=True).hidden_states).squeeze(1).to(self.compute_dtype)
            y = torch.stack(self.alt_model(tokens, output_hidden_states=True).hidden_states).squeeze(1).to(self.compute_dtype)
            x, y = x[:, 1:, :], y[:, 1:, :]
            x = torch.cat((x, torch.ones_like(x[:, :, :1])), dim=2)
        return x,y
    
    def train(self, train_dataset: Dataset, valid_ds: Dataset, reg_param: float=1e-6):
        torch.set_default_dtype(self.compute_dtype)
        with torch.device(self.device):
            y_mean = torch.zeros((self.layers, self.d))
            num_tokens = 0
            for example in valid_ds: 
                y = self.get_x_y(example)[1]
                num_tokens += y.shape[1]
                y_mean += y.sum(dim=1)
            y_mean /= num_tokens
            xtx = torch.zeros((self.layers, self.d+1, self.d+1))
            xty = torch.zeros((self.layers, self.d+1, self.d))
            for example in train_dataset:
                x, y = self.get_x_y(example)
                xtx += (x.transpose(-1,-2) @ x)
                xty += (x.transpose(-1,-2) @ y)
            reg = reg_param * xtx.diagonal(dim2=-2, dim1=-1)
            reg[:, -1] = 0
            xtx += torch.diag_embed(reg)
            w = torch.linalg.solve(xtx, xty)
            pred_error_sum = torch.zeros((self.layers))
            standard_error_sum = torch.zeros((self.layers))
            for example in valid_ds:
                x,y = self.get_x_y(example)
                pred_error_sum += ((x@w - y) ** 2).sum(dim=(1,2))
                standard_error_sum += ((y_mean.unsqueeze(1) - y) ** 2).sum(dim=(1,2))
            unexplained_var = pred_error_sum / standard_error_sum
            torch.save(w, "/scratch/an4245/linear_translator_w.pt")
        return unexplained_var

if __name__=="__main__": 
    ds = load_from_disk("/scratch/an4245/webtext").with_format("torch", device="cuda")
    print(Translate().train(ds.select(range(400)), ds.select(range(400,500))))


   
                 