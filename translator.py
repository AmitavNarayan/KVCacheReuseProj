import wandb
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from datasets import Dataset
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
    def __init__(self, model: AutoModelForCausalLM, alt_model: AutoModelForCausalLM):
        self.model = model
        self.alt_model = alt_model
        self.model.eval()
        self.alt_model.eval()
        self.device = self.model.device
        self.translator = Translator(self.model.config.num_hidden_layers+1, model.config.hidden_size, alt_model.config.hidden_size, self.device, dtype=torch.float32)
        self.optim = torch.optim.AdamW(self.translator.parameters())
        
    def forward_pass(self, example: dict):
        with torch.no_grad():
            prompt = torch.cat((example["prefix"], torch.cat(example["docs"]), example["suffix"], example["answer"])).unsqueeze(0)
            x = torch.stack(self.model(prompt, output_hidden_states=True).hidden_states).squeeze(1).to(self.translator.dtype)
            y = torch.stack(self.alt_model(prompt, output_hidden_states=True).hidden_states).squeeze(1).to(self.translator.dtype)
        layer_loss = einx.mean("b [...]", (self.translator(x) - y) ** 2)
        return layer_loss

    def record_loss(self, layer_loss: torch.Tensor, loss_type: str, step: int):
        loss_dict = {f"{loss_type}-loss": layer_loss.mean().item()}
        for l in range(self.model.config.num_hidden_layers+1):
            loss_dict[f"{loss_type}-loss-layer-{l}"] = layer_loss[l].item()
        wandb.log(loss_dict, step=step)
        
    
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, step_size: int = 64, valid_size: int = 100, valid_freq: float = 0.025):
        wandb.init()
        self.translator.train()
        loss = 0
        step = 1
        valid_ds = valid_dataset.select(range(valid_size))
        for example in train_dataset:
            layer_loss = self.forward_pass(example) 
            self.record_loss(layer_loss, "train", step)
            loss = layer_loss.mean()
            loss = loss / step_size
            loss.backward()
            if step % step_size == 0:
                self.optim.step()
                self.optim.zero_grad()
            if step % int(len(train_dataset) * valid_freq) == 0:
                with torch.no_grad():
                    layer_loss = 0
                    for example in valid_ds:
                        layer_loss += self.forward_pass(example)
                    layer_loss = layer_loss / valid_size
                    self.record_loss(layer_loss, "valid", step)
            step += 1
        wandb.finish()


                 