

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
import torch.nn.functional as F
from datasets import Dataset
from hotpot_evaluate_v1 import f1_score

class Packetize(torch.nn.Module):
    def __init__(self, device: str | torch.device, d_model: int, header_len: int, mean: float = 0, stdev: float = 0.02):
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
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, header_len: int = 8, lr: float=1e-3, max_new_tokens: int=64):
        self.device = model.device
        self.model = model
        self.embed = self.model.get_input_embeddings()
        self.model.generation_config.max_new_tokens=max_new_tokens
        for param in self.model.parameters(): param.requires_grad=False
        self.packetize = Packetize(self.device, self.model.config.hidden_size, header_len)
        self.optimizer = torch.optim.AdamW(self.packetize.parameters(), lr=lr)
        self.tokenizer = tokenizer 
       
    def get_attn_mask(self, pos: torch.Tensor):
        positions = pos
        row_mask = torch.arange(positions[0], positions[-2], device=self.device).unsqueeze(1)
        col_mask = torch.arange(0, positions[-2], device=self.device).unsqueeze(0) < positions[torch.searchsorted(positions, row_mask, right=True)-1]
        attn_mask = torch.tril(torch.ones((positions[-1], positions[-1]), device=self.device, dtype=torch.bool))
        attn_mask[positions[0]:positions[-2], :positions[-2]][col_mask] = False
        return attn_mask

    def get_prompt_mask(self, example: dict, teacher_output_embeds: torch.Tensor = None):
        lens = [example["prefix"].shape[0], *[doc.shape[0] for doc in example["docs"]], example["suffix"].shape[0]]
        student_prompt = torch.cat((example["prefix"], torch.cat(example["docs"]), example["suffix"]))
        if teacher_output_embeds is not None: 
            lens[-1] += teacher_output_embeds.shape[0]
            student_prompt = torch.cat((student_prompt, teacher_output_embeds))
        attn_mask = self.get_attn_mask(torch.tensor(lens, device=self.device).cumsum(0))
        return student_prompt, attn_mask

 
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, step_size: int=32, valid_freq:float=0.025, valid_size: int=100):
        wandb.init()
        self.model.train()
        train_examples = train_dataset.shuffle()
        valid_examples = valid_dataset.shuffle().select(range(valid_size))
        step = 1
        for example in train_examples:
            example["prefix"] = self.embed(example["prefix"])
            example["suffix"] = self.embed(example["suffix"])
            example["docs"] = [self.embed(doc) for doc in example["docs"]]
            teacher_prompt = torch.cat((example["prefix"], torch.cat(example["docs"]), example["suffix"]))
            teacher_output = self.model.generate(inputs_embeds = teacher_prompt.unsqueeze(0), do_sample=False, return_dict_in_generate=True, output_logits=True)
            teacher_output_logits = torch.stack(teacher_output.logits, dim=1)[0]
            teacher_output_embeds = self.embed(teacher_output.sequences[0])

            example["docs"] = [self.packetize(doc) for doc in example["docs"]]
            student_prompt, attn_mask = self.get_prompt_mask(example, teacher_output_embeds)
            student_output_logits = self.model(inputs_embeds=student_prompt.unsqueeze(0), attention_mask=attn_mask.reshape(1,1,attn_mask.shape[0], -1), logits_to_keep=teacher_output_embeds.shape[0]+1).logits
            student_output_logits = student_output_logits[0][-teacher_output_embeds.shape[0]-1:-1, :]
            
            loss = F.kl_div(F.log_softmax(student_output_logits, dim=1), F.log_softmax(teacher_output_logits, dim=1), log_target=True, reduction="batchmean")
            wandb.log({"train-loss": loss.item()})
            loss = loss / step_size
            loss.backward()
            if step % step_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            if step % int(valid_freq * len(train_dataset)) == 0:
                self.model.eval()
                with torch.no_grad():
                    score = 0
                    for valid_example in valid_examples:
                        output_tokens = []
                        valid_example["prefix"] = self.embed(valid_example["prefix"])
                        valid_example["suffix"] = self.embed(valid_example["suffix"])
                        valid_example["docs"] = [self.packetize(self.embed(doc)) for doc in valid_example["docs"]]
                        answer = valid_example["answer"]
                        prompt, mask = self.get_prompt_mask(valid_example)
                        output = self.model(inputs_embeds=prompt.unsqueeze(0), attention_mask=mask.reshape(1,1, mask.shape[0], -1), logits_to_keep=1, use_cache=True)
                        while len(output_tokens) < self.model.generation_config.max_new_tokens:
                            output_tokens.append(output.logits[0, -1].argmax())
                            if output_tokens[-1]==self.tokenizer.eos_token_id:
                                break
                            output_embed = self.embed(output_tokens[-1].reshape(1,1))
                            output = self.model(inputs_embeds = output_embed, past_key_values=output.past_key_values, use_cache=True)
                        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                        answer_text = self.tokenizer.decode(answer, skip_special_tokens=True)
                        score += f1_score(output_text, answer_text)[0] 
                    score = score / len(valid_examples)
                    wandb.log({"f1-score": score})
                self.model.train()
            step += 1
        wandb.finish()
        return
