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
                 alt_model_path: str = "Qwen2.5-Coder-1.5B-Instruct", model_dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=model_dtype).to(self.device)
        self.alt_model = AutoModelForCausalLM.from_pretrained(alt_model_path, dtype=model_dtype).to(self.device)
        self.layers = self.model.config.num_hidden_layers + 1
        self.model.eval()
        self.alt_model.eval()
        self.device = self.model.device
        self.translator = Translator(self.layers, self.model.config.hidden_size, self.alt_model.config.hidden_size, self.device, dtype=torch.float32)
        self.optim = torch.optim.AdamW(self.translator.parameters())

    def get_x_y(self, example: dict):
        with torch.no_grad():
            prompt = torch.cat((example["prefix"], torch.cat(list(example["docs"])), example["suffix"], example["answer"])).unsqueeze(0)
            x = torch.stack(self.model(prompt, output_hidden_states=True).hidden_states).squeeze(1).to(self.translator.dtype)
            y = torch.stack(self.alt_model(prompt, output_hidden_states=True).hidden_states).squeeze(1).to(self.translator.dtype)
        return x, y
    
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, step_size: int = 64, valid_size: int = 100, valid_freq: float = 0.01):
        wandb.init()
        self.translator.train()
        valid_ds = valid_dataset.select(range(valid_size))
        y_mean = torch.zeros((self.layers, self.alt_model.config.hidden_size), device=self.device, dtype=torch.float64)
        num_tokens = 0
        for example in valid_ds: 
            y = self.get_x_y(example)[1]
            num_tokens += y.shape[1]
            y_mean += y.sum(dim=1)
        y_mean /= num_tokens
        loss = 0
        step = 1
        for example in train_dataset:
            x, y = self.get_x_y(example)
            loss = ((self.translator(x) - y) ** 2).sum() / step_size
            loss.backward()
            if step % step_size == 0:
                self.optim.step()
                self.optim.zero_grad()
            if step % int(len(train_dataset) * valid_freq) == 0:
                with torch.no_grad():
                    pred_error_sum = torch.zeros((self.layers), device=self.device, dtype=torch.float64)
                    standard_error_sum = torch.zeros((self.layers), device=self.device, dtype=torch.float64)
                    for example in valid_ds:
                        x,y = self.get_x_y(example)
                        pred_error_sum += ((self.translator(x) - y) ** 2).sum(dim=(1,2))
                        standard_error_sum += ((y_mean.unsqueeze(1) - y) ** 2).sum(dim=(1,2))
                    unexplained_var = pred_error_sum / standard_error_sum
                    valid_loss = {f"layer-{l}-loss": unexplained_var[l].item() for l in range(self.layers)}
                    wandb.log(valid_loss)
            step += 1
        wandb.finish()

if __name__=="__main__":
    ds = load_from_disk("hotpot").with_format("torch", device="cuda")
    Translate().train(ds["hotpot_train"], ds["hotpot_valid"])
    
                 