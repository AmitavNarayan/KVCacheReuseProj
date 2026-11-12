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

    def validate(self, valid_inputs: Dataset, is_alt: bool):
        if is_alt: self.alt_model.eval()
        else: self.model.eval()
        with torch.no_grad():
            score = 0
            for info in valid_inputs:
                prompt, mask, answer = info["prompt"], info["mask"], info["answer"]
                with torch.autocast(device_type="cuda", dtype=self.compute_dtype): 
                    if is_alt: output = self.alt_model(prompt.reshape(1,-1), attention_mask=mask.reshape(1,1,mask.shape[0], -1), logits_to_keep=1, use_cache=True)
                    else: output = self.model(prompt.reshape(1,-1), attention_mask=mask.reshape(1,1,mask.shape[0], -1), logits_to_keep=1, use_cache=True)
                output_tokens = []
                while len(output_tokens) < self.model.generation_config.max_new_tokens:
                    output_tokens.append(output.logits[0, -1].argmax())
                    if output_tokens[-1]==self.tokenizer.eos_token_id:
                        break
                    with torch.autocast(device_type="cuda", dtype=self.compute_dtype): 
                        if is_alt: output = self.alt_model(output_tokens[-1].reshape(1,1), past_key_values=output.past_key_values, use_cache=True)
                        
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                answer_text = self.tokenizer.decode(answer, skip_special_tokens=True)
                score += f1_score(output_text, answer_text)[0] 
            score = score / len(valid_inputs)
            wandb.log({"f1-score": score})
        if is_alt: self.alt_model.train()
        else: self.model.train()
    
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
                f1 = self.validate(valid_inputs, is_alt=False)
                alt_f1 = self.validate(valid_inputs, is_alt=True)
            step += 1
        wandb.finish()
        return




    


class KVLink():
    def __init__(self, model_path: str = "Qwen2.5-1.5B-Instruct", 
                 alt_model_path: str = "Qwen2.5-Coder-1.5B-Instruct", 
                 tokenizer_path: str = "Qwen2.5-1.5B-Instruct", 
                 model_dtype: torch.dtype = torch.bfloat16, 
                 device: torch.device | str = "cuda", 
                 header_len: int = 8, 
                 lr: float=1e-3, 
                 max_new_tokens: int=64):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=model_dtype).to(self.device)
        self.alt_model = AutoModelForCausalLM.from_pretrained(alt_model_path, dtype=model_dtype).to(self.device)
        self.model.config.use_cache = True
        self.alt_model.config.use_cache = True
        self.max_new_tokens = max_new_tokens
        self.model.requires_grad_(False)
        self.packetize = Packetize(self.device, self.model.config.hidden_size, header_len)
        self.optimizer = torch.optim.AdamW(self.packetize.parameters(), lr=lr)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.vllm = LLM(model_path, dtype=model_dtype, gpu_memory_utilization=0.3, enable_sleep_mode=True)
       

    def validate(self, valid_examples: Dataset, is_alt: bool = False):
        write_model = self.alt_model if is_alt else self.model
        write_embed = write_model.get_input_embeddings()
        self.model.eval()  
        self.alt_model.eval()  
        with torch.no_grad():
            score = 0
            for valid_example in valid_examples:
                packetized_docs = [self.packetize(write_embed(doc)) for doc in valid_example["docs"]]
                doc_mask = torch.block_diag(*[torch.ones((pd.shape[0], pd.shape[0]), dtype=torch.bool, device=self.device).tril() for pd in packetized_docs])[None, None]
                cache = self.model(input_ids = valid_example["prefix"][None]).past_key_values
                pos_ids = valid_example["prefix"].shape[0] + torch.arange(doc_mask.shape[-1], device=self.device)
                doc_cache = write_model(inputs_embeds = torch.cat(packetized_docs)[None], attention_mask=doc_mask, position_ids=pos_ids[None]).past_key_values
                for i, layer in enumerate(doc_cache.layers): cache.update(layer.keys, layer.values, i)
                output_tokens = []
                output = self.model(input_ids = valid_example["suffix"][None], past_key_values=cache, logits_to_keep=1)
                for _ in range(self.max_new_tokens):
                    output_tokens.append(output.logits[0, 0].argmax())
                    if output_tokens[-1]==self.tokenizer.eos_token_id:
                        break
                    output = self.model(input_ids = output_tokens[-1].reshape(1,1), past_key_values=output.past_key_values, logits_to_keep=1)
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                answer_text = self.tokenizer.decode(valid_example["answer"], skip_special_tokens=True)
                score += f1_score(output_text, answer_text)[0] 

            score = score / len(valid_examples)
        self.model.train()
        return score
 
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, teacher_batch_size: int=32, step_size: int=32, valid_freq:float=0.01, valid_size: int=100):
        wandb.init()
        self.model.train()
        valid_examples = valid_dataset.select(range(valid_size))
        t_output = self.vllm.generate([{"prompt_token_ids": torch.cat([example["prefix"], *example["docs"], example["suffix"]])} for example in train_dataset.with_format(None)], SamplingParams(temperature=0, max_tokens=self.max_new_tokens))
        self.vllm.sleep(level=2)
        t_output_tokens = [torch.tensor(x.outputs[0].token_ids, device=self.device) for x in t_output]
        for step, example in enumerate(train_dataset, start=1):
            embed = self.model.get_input_embeddings()
            answer = t_output_tokens[step-1]
            t_tokens = torch.cat((example["prefix"], *example["docs"], example["suffix"], answer))
            t_output_logits = self.model(input_ids=t_tokens[None], logits_to_keep=len(answer)+1).logits[0, :-1]
            t_output_embeds = embed(answer)

            segments = [embed(example["prefix"]), *[self.packetize(embed(doc)) for doc in example["docs"]], torch.cat((embed(example["suffix"]), t_output_embeds))]
            mask = torch.block_diag(*[torch.ones((s.shape[0], s.shape[0]), device=self.device, dtype=torch.bool).tril() for s in segments])
            mask[-segments[-1].shape[0]:, :-segments[-1].shape[0]] = 1
            s_output_logits = self.model(inputs_embeds=torch.cat(segments)[None], attention_mask=mask[None, None], logits_to_keep=len(answer)+1).logits[0, :-1]
            
            loss = F.kl_div(F.log_softmax(s_output_logits, dim=1), F.log_softmax(t_output_logits, dim=1), log_target=True, reduction="batchmean")
            wandb.log({"train-loss": loss.item()})
            (loss / step_size).backward()
            if step % step_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            if step % int(valid_freq * len(train_dataset)) == 0:
                f1 = self.validate(valid_examples)
                wandb.log({"f1-score": f1})
                alt_f1 = self.validate(valid_examples, is_alt=True)
                wandb.log({"alt-f1-score": alt_f1})
                
        wandb.finish()
