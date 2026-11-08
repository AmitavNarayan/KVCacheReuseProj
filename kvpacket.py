

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Packetize(torch.nn.Module):
    def __init__(self, device: str, d_model: int, header_len: int, mean: float = 0, stdev: float = 0.02):
        super().__init__()
        self.header_len = header_len
        self.header = torch.nn.Parameter(torch.zeros((header_len, d_model), device=device))
        self.trailer = torch.nn.Parameter(torch.zeros((header_len, d_model), device=device))
        torch.nn.init.normal_(self.header, mean=mean, std=stdev)
        torch.nn.init.normal_(self.trailer, mean=mean, std=stdev)
    def forward(self, x: torch.Tensor):
        new_x = torch.cat((self.header.to(x.dtype), x, self.trailer.to(x.dtype)), dim=-2)
        return new_x 

class KVPacket():
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, header_len: int = 8, lr: float=5e-2, max_new_tokens: int=64):
        self.device = model.device
        self.model = model
        self.model.generation_config.max_new_tokens=max_new_tokens
        for param in self.model.parameters(): param.requires_grad=False
        self.packetize = Packetize(self.device, self.model.config.hidden_size, header_len)
        self.optimizer = torch.optim.AdamW(self.packetize.parameters(), lr=lr)
        self.tokenizer = tokenizer 
        self.header_len = header_len 
       
    def get_attn_mask(self, pos: torch.Tensor):
        positions = pos
        row_mask = torch.arange(positions[0], positions[-2], device=self.device).unsqueeze(1)
        col_mask = torch.arange(0, positions[-2], device=self.device).unsqueeze(0) < positions[torch.searchsorted(positions, row_mask, right=True)-1]
        attn_mask = torch.tril(torch.ones((positions[-1], positions[-1]), device=self.device, dtype=torch.bool))
        attn_mask[positions[0]:positions[-2], :positions[-2]][col_mask] = False
        return attn_mask
 
    def train(self, dataset: dict[str, torch.Tensor], step_size: int=32):
        wandb.init()
        self.model.train()
        loader = DataLoader(dataset, batch_size=None, shuffle=True)
        step = 1
        for example in loader:
            embed = self.model.get_input_embeddings()
            doc_embeds = []
            for key in example: 
                if key=="docs": 
                    for i, doc in enumerate(example["docs"]): 
                        doc_embeds.append(embed(doc.to(self.device)))
                        example["docs"][i] = self.packetize(doc_embeds[-1])
                else: 
                    example[key] = embed(example[key].to(self.device))

            doc_embeds = torch.cat(doc_embeds)
            teacher_prompt = torch.cat((example["prefix"], doc_embeds, example["suffix"]))
            teacher_output = self.model.generate(inputs_embeds = teacher_prompt[None, :], do_sample=False, return_dict_in_generate=True, output_logits=True)
            teacher_output_logits = torch.stack(teacher_output.logits, dim=1)[0]
            teacher_output_embeds = embed(teacher_output.sequences[0])

            lens = [example["prefix"].shape[0], *[doc.shape[0] for doc in example["docs"]], example["suffix"].shape[0]+teacher_output_embeds.shape[0]]
            attn_mask = self.get_attn_mask(torch.tensor(lens, device=self.device).cumsum(0))
            student_prompt = torch.cat((example["prefix"], torch.cat(example["docs"]), example["suffix"], teacher_output_embeds))
            student_output_logits = self.model(inputs_embeds=student_prompt[None, :], attention_mask=attn_mask[None, None, :, :]).logits
            student_output_logits = student_output_logits[0][-teacher_output_embeds.shape[0]-1:-1, :]
            
            loss = F.kl_div(F.log_softmax(student_output_logits, dim=1), F.log_softmax(teacher_output_logits, dim=1), log_target=True, reduction="batchmean")
            wandb.log({"train-loss": loss.item()})
            loss = loss / step_size
            loss.backward()
            if step % step_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            step += 1
        wandb.finish()
        return

    
    







    
    
    
    
        