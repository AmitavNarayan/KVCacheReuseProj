from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from hotpot_evaluate_v1 import f1_score
from datasets import Dataset, load_from_disk
import torch.nn.functional as F 


class KVLink():
    def __init__(self, max_num_docs: int=20, num_link_tokens: int = 5, max_new_tokens: int = 64, compute_dtype=torch.bfloat16, model_path: str = "Qwen2.5-1.5B-Instruct", 
                 alt_model_path: str = "Qwen2.5-Coder-1.5B-Instruct", 
                 tokenizer_path: str = "Qwen2.5-1.5B-Instruct", 
                 model_dtype: torch.dtype = torch.float32, 
                 device: torch.device | str = "cuda", 
                 lr: float=5e-6):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=model_dtype).to(self.device)
        self.alt_model = AutoModelForCausalLM.from_pretrained(alt_model_path, dtype=model_dtype).to(self.device)
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = True
        self.alt_model.config.use_cache = True
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.link_tokens = [f"<link{a+1}_{b+1}>" for a in range(max_num_docs) for b in range(num_link_tokens)]
        self.num_link_tokens = num_link_tokens
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.link_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.link_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.link_tokens), device=self.device, dtype=torch.long).reshape(max_num_docs, num_link_tokens)
        self.compute_dtype = compute_dtype
        
    def validate(self, valid_examples: Dataset, is_alt: bool = False):
        write_model = self.alt_model if is_alt else self.model
        self.model.eval()  
        self.alt_model.eval()  
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.compute_dtype):
            score = 0
            for valid_example in valid_examples:
                answer_len = example["answer"].shape[0]
                segments = [*[torch.cat((doc, self.link_ids[i, :])), example["prefix"] for i, doc in enumerate(example["docs"])], torch.cat((example["suffix"], example["answer"]))]
                prompt = torch.cat(segments)
                mask = torch.block_diag(*[torch.ones((s.shape[0], s.shape[0]), device=self.device, dtype=torch.bool).tril() for s in segments])
                idx = example["prefix"].shape[0]
                for doc in example["docs"]: 
                    mask[idx+doc.shape[0]:idx+doc.shape[0]+self.num_link_tokens, :idx+doc.shape[0]]=1
                    idx += doc.shape[0] + self.num_link_tokens
                mask[-segments[-1].shape[0]:, :-segments[-1].shape[0]] = 1
                with torch.autocast(device_type=self.device, dtype=self.compute_dtype): 
                    output_logits = self.model(input_ids=prompt[None], attention_mask=mask[None, None], logits_to_keep=answer_len+1).logits[0, :-1]
                    loss = F.cross_entropy(output_logits, prompt[-answer_len:])
                wandb.log({"train-loss": loss.item()})
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                answer_text = self.tokenizer.decode(valid_example["answer"], skip_special_tokens=True)
                score += f1_score(output_text, answer_text)[0] 

            score = score / len(valid_examples)
        self.model.train()
        return score
    
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, step_size: int=32, valid_freq:float=0.01, valid_size: int=100):
        wandb.init()
        self.model.train()
        valid_examples = valid_dataset.select(range(valid_size))
        for step, example in enumerate(train_dataset, start=1):

            answer_len = example["answer"].shape[0]
            segments = [example["prefix"], *[torch.cat((doc, self.link_ids[i, :])) for i, doc in enumerate(example["docs"])], torch.cat((example["suffix"], example["answer"]))]
            prompt = torch.cat(segments)
            mask = torch.block_diag(*[torch.ones((s.shape[0], s.shape[0]), device=self.device, dtype=torch.bool).tril() for s in segments])
            idx = example["prefix"].shape[0]
            for doc in example["docs"]: 
                mask[idx+doc.shape[0]:idx+doc.shape[0]+self.num_link_tokens, :idx+doc.shape[0]]=1
                idx += doc.shape[0] + self.num_link_tokens
            mask[-segments[-1].shape[0]:, :-segments[-1].shape[0]] = 1
            with torch.autocast(device_type=self.device, dtype=self.compute_dtype): 
                output_logits = self.model(input_ids=prompt[None], attention_mask=mask[None, None], logits_to_keep=answer_len+1).logits[0, :-1]
                loss = F.cross_entropy(output_logits, prompt[-answer_len:])
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

if __name__=="__main__":
    ds = load_from_disk("hotpot").with_format("torch", device="cuda")
    KVLink().train(ds["hotpot_train"], ds["hotpot_valid"])


