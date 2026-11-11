from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from hotpot_evaluate_v1 import f1_score
from datasets import Dataset
import torch.nn.functional as F

class KVLink():
    def __init__(self, model: AutoModelForCausalLM, alt_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_num_docs: int=20, num_link_tokens: int = 5, lr: float=5e-6, max_new_tokens: int = 64, compute_dtype=torch.bfloat16):
        self.device = model.device
        self.model = model
        self.alt_model = alt_model
        self.model.generation_config.max_new_tokens=max_new_tokens
        self.model.gradient_checkpointing_enable()
        self.tokenizer = tokenizer
        self.num_link_tokens = num_link_tokens
        self.link_tokens = [f"<link{a+1}_{b+1}>" for a in range(max_num_docs) for b in range(num_link_tokens)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.link_tokens})
        self.model.resize_token_embeddings(len(tokenizer))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.link_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.link_tokens), device=self.device, dtype=torch.long).reshape(max_num_docs, num_link_tokens)
        self.compute_dtype = compute_dtype

    def get_attn_mask(self, positions: torch.Tensor):
        row_mask = torch.arange(positions[0], positions[-3], device=self.device).unsqueeze(1)
        col_mask = torch.arange(0, positions[-3], device=self.device).unsqueeze(0)
        col_mask = col_mask < positions[torch.searchsorted(positions, row_mask, right=True)-1]
        row_mask = self.num_link_tokens < positions[torch.searchsorted(positions, row_mask, right=True)] - row_mask
        attn_mask = torch.tril(torch.ones((positions[-1], positions[-1]), device=self.device, dtype=torch.bool))
        attn_mask[positions[0]:positions[-3], :positions[-3]][col_mask & row_mask] = False
        return attn_mask

    def get_prompt_pos(self, example: dict):
        lens = [example["prefix"].shape[0], *[doc.shape[0]+self.num_link_tokens for doc in example["docs"]], example["suffix"].shape[0], example["answer"].shape[0]]
        example["docs"] = torch.cat([torch.cat((doc, self.link_ids[i, :])) for i, doc in enumerate(example["docs"])])
        prompt = torch.cat((example["prefix"], example["docs"], example["suffix"], example["answer"]))
        pos = torch.tensor(lens, device=self.device).cumsum(0)
        return prompt, pos
    
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, step_size:int=32, valid_freq:float=0.01, valid_size: int=100):
        wandb.init()
        self.model.train()
        train_examples = train_dataset
        valid_examples = valid_dataset.select(range(valid_size))
        valid_inputs = []
        for example in valid_examples:
            prompt, pos = self.get_prompt_pos(example)
            answer_pos = pos[-2]
            info = {}
            info["prompt"] = prompt[:answer_pos]
            info["mask"] = self.get_attn_mask(pos)[:answer_pos, :answer_pos]
            info["answer"] = prompt[answer_pos:]
            valid_inputs.append(info)
        step = 1
        for example in train_examples:
            prompt, pos = self.get_prompt_pos(example)
            attn_mask = self.get_attn_mask(pos)
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype): 
                answer_len = example["answer"].shape[0]
                output_logits = self.model(prompt.reshape(1,-1), attention_mask=attn_mask.reshape(1,1,attn_mask.shape[0],-1), logits_to_keep = answer_len + 1).logits[0, :-1]
                loss = F.cross_entropy(output_logits, prompt[-answer_len:])
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
                    for info in valid_inputs:
                        prompt, mask, answer = info["prompt"], info["mask"], info["answer"]
                        with torch.autocast(device_type="cuda", dtype=self.compute_dtype): 
                            output = self.alt_model(prompt.reshape(1,-1), attention_mask=mask.reshape(1,1,mask.shape[0], -1), logits_to_keep=1, use_cache=True)
                        output_tokens = []
                        while len(output_tokens) < self.model.generation_config.max_new_tokens:
                            output_tokens.append(output.logits[0, -1].argmax())
                            if output_tokens[-1]==self.tokenizer.eos_token_id:
                                break
                            with torch.autocast(device_type="cuda", dtype=self.compute_dtype): 
                                output = self.alt_model(output_tokens[-1].reshape(1,1), past_key_values=output.past_key_values, use_cache=True)
                        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                        answer_text = self.tokenizer.decode(answer, skip_special_tokens=True)
                        score += f1_score(output_text, answer_text)[0] 
                    score = score / len(valid_inputs)
                    wandb.log({"f1-score": score})
                self.model.train()
            step += 1
        wandb.finish()
        return