

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from torch.utils.data import DataLoader
import einx

class Packetize(torch.nn.Module):
    def __init__(self, device: str, d_model: int, header_len: int, mean: float = 0, stdev: float = 0.02):
        super().__init__()
        self.header_len = header_len
        self.header = torch.nn.Parameter(torch.zeros((header_len, d_model), device=device))
        self.trailer = torch.nn.Parameter(torch.zeros((header_len, d_model), device=device))
        torch.nn.init.normal_(self.header, mean=mean, std=stdev)
        torch.nn.init.normal_(self.trailer, mean=mean, std=stdev)
    def forward(self, x: torch.Tensor):
        new_x = einx.id("X D, ... T D, Y D-> ... (X + T + Y) D", self.header.to(x.dtype), x, self.trailer.to(x.dtype))
        return new_x 

class KVPacket():
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, header_len: int = 8, num_docs: int = 10, lr: float=5e-4, max_new_tokens: int=64):
        self.device = model.device
        self.model = model
        self.model.generation_config.max_new_tokens=max_new_tokens
        for param in self.model.parameters(): param.requires_grad=False
        self.packetize = Packetize(self.device, self.model.config.hidden_size, header_len)
        self.optimizer = torch.optim.AdamW(self.packetize.parameters(), lr=lr)
        self.tokenizer = tokenizer
        self.num_docs = num_docs   
        self.header_len = header_len 
       

    def get_attn_mask(self, prompt_len: int, prefix_len: int, doc_len: int, batch_size: int):
            row_mask = torch.arange(prompt_len, device=self.device).unsqueeze(1)
            col_mask = torch.arange(prompt_len, device=self.device).unsqueeze(0)
            attn_mask = torch.tril(torch.ones((prompt_len, prompt_len), device=self.device, dtype=bool))
            col_mask = col_mask < (prefix_len + ((row_mask - prefix_len) // doc_len) * doc_len)
            row_mask = (row_mask >= prefix_len) & (row_mask < (prefix_len + self.num_docs * doc_len))
            attn_mask[col_mask & row_mask] = False
            attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(self.model.dtype)
            attn_mask = einx.id("... -> B 1 ...", attn_mask, B=batch_size)
    
            return attn_mask

    def get_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, teacher_padmask: torch.Tensor):
        student_probs = student_logits.float().log_softmax(dim=-1)
        teacher_probs = teacher_logits.float().log_softmax(dim=-1)
        kldivloss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        loss = kldivloss(student_probs, teacher_probs).sum(-1)
        true_loss = (loss * teacher_padmask).sum() / teacher_padmask.sum()
        return true_loss
    
    def run(self, prefix_tokens: torch.Tensor, doc_tokens: torch.Tensor, suffix_tokens: torch.Tensor):
        prefix_embeds = self.model.get_input_embeddings()(prefix_tokens)
        doc_embeds = self.packetize(self.model.get_input_embeddings()(doc_tokens))
        suffix_embeds = self.model.get_input_embeddings()(suffix_tokens)
        prompt_embeds = einx.id("B T1 D, B X T2 D, B T3 D -> B (T1 + (X T2) + T3) D", prefix_embeds, doc_embeds, suffix_embeds)
        prompt_tokens = einx.id("B T1, B X T2, B T3-> B (T1 + (X T2) + T3)", prefix_tokens, doc_tokens, suffix_tokens)
        prompt_len = prompt_tokens.shape[1]
        
        attn_mask = self.get_attn_mask(prompt_embeds.shape[1], prefix_embeds.shape[1], doc_embeds.shape[2], prefix_embeds.shape[0])
        with torch.no_grad():
            teacher_answer = self.model.generate(prompt_tokens, return_dict_in_generate=True, output_logits=True, do_sample=False)
            teacher_answer_tokens = teacher_answer.sequences[:, prompt_len:]
            teacher_answer_padmask = teacher_answer_tokens != self.tokenizer.pad_token_id
            teacher_answer_logits = torch.stack(teacher_answer.logits, dim=1)
        student_answer_first = self.model(inputs_embeds=prompt_embeds, attention_mask=attn_mask, logits_to_keep=1)
        student_answer_full = self.model(input_ids=teacher_answer_tokens, past_key_values=student_answer_first.past_key_values) 
        student_answer_logits = einx.id("B 1 V, B T V -> B (1+T) V", student_answer_first.logits, student_answer_full.logits[:, :-1, :])

        loss = self.get_loss(student_answer_logits, teacher_answer_logits, teacher_answer_padmask)
        return loss
    
    def train(self, dataloader: DataLoader):
        wandb.init()
        self.model.train()
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(self.device)
            loss = self.run(batch["prefix"], batch["docs"], batch["suffix"])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            wandb.log({"train-loss": loss.item()})
        wandb.finish()
        return self.model, self.optimizer
    



    




    