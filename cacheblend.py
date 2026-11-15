from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import wandb
from hotpot_evaluate_v1 import f1_score
from datasets import Dataset, load_from_disk
import torch.nn.functional as F 


class CacheBlend():
    def __init__(self, max_num_docs: int=20, max_new_tokens: int = 64, compute_dtype=torch.bfloat16, model_path: str = "Qwen2.5-1.5B-Instruct", 
                 alt_model_path: str = "Qwen2.5-Coder-1.5B-Instruct", 
                 tokenizer_path: str = "Qwen2.5-1.5B-Instruct", 
                 model_dtype: torch.dtype = torch.float32, 
                 alt_model_dtype: torch.dtype = torch.bfloat16,
                 device: torch.device | str = "cuda",
                 lr: float=5e-6):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=model_dtype).to(self.device)
        config = AutoConfig.from_pretrained(model_path)
        config.num_hidden_layers = 2
        self.trunc_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, dtype=model_dtype).to(self.device)
        self.alt_model = AutoModelForCausalLM.from_pretrained(alt_model_path, dtype=alt_model_dtype).to(self.device)
        self.alt_model.requires_grad_(False)
        self.model.gradient_checkpointing_enable()
        self.max_new_tokens = max_new_tokens
        self.layers = self.model.config.num_hidden_layers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.compute_dtype = compute_dtype
        
    def validate(self, r: float, valid_examples: Dataset, is_alt: bool = False):
        write_model = self.alt_model if is_alt else self.model
        self.model.eval()
        f1 = 0
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.compute_dtype), torch.device(self.device):
            for valid_example in valid_examples:
                p = valid_example["prefix"].shape[0]
                s = valid_example["suffix"].shape[0]
                docs = valid_example["docs"]
                doc_pos = torch.tensor([], dtype=torch.long)
                offset = p
                for doc in docs: 
                    doc_pos = torch.cat((doc_pos, offset + torch.arange(doc.shape[0])), dim=0)
                    offset += doc.shape[0]
                doc_mask = torch.block_diag(*[torch.ones(d.shape[0], d.shape[0]).tril() for d in docs]).bool()[None, None]
                doc_cache = write_model(input_ids=torch.cat(docs)[None], position_ids=doc_pos[None], attention_mask=doc_mask).past_key_values
                cache = self.model(input_ids=valid_example["prefix"][None]).past_key_values
                for l, layer in enumerate(doc_cache.layers):
                    cache.update(layer.keys, layer.values, l) 
                true_cache = self.trunc_model(input_ids=torch.cat((valid_example["prefix"], *docs))[None]).past_key_values
                dist_k, dist_v = [torch.linalg.vector_norm(getattr(true_cache.layers[1], x)[:, :, p:].float()-getattr(cache.layers[1], x)[:, :, p:].float(), dim=(0,1,3)) for x in ["keys", "values"]]
                score = dist_k / dist_k.mean() + dist_v / dist_v.mean()
                tok_pos = score.topk(int(r * score.shape[0])).indices.sort().values

                
                for _ in range(self.max_new_tokens):
                    output = self.model(input_ids=ls, attention_mask = ls_mask, past_key_values=cache, position_ids=p+toks, logits_to_keep=1)
                    output_tokens.append(output.logits[0, -1].argmax())
                    if output_tokens[-1] == self.tokenizer.eos_token_id:
                        break
                    ls, ls_mask, ls_ids = output_tokens[-1][None, None], None, (ls_ids[:, -1]+1)[None]

                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                answer_text = self.tokenizer.decode(valid_example["answer"], skip_special_tokens=True)
                f1 += f1_score(output_text, answer_text)[0]
            return f1 / len(valid_examples)
    
    def run(self, valid_dataset: Dataset, checkpt_freq: float = 0.1, valid_size: int=100, r: float = 0.15):
        wandb.init()
        valid_examples = valid_dataset.select(range(valid_size))
        for step, example in enumerate(valid_examples):
            f1 = self.validate(valid_examples)
            wandb.log({"f1-score": f1})
            alt_f1 = self.validate(valid_examples, is_alt=True)
            wandb.log({"alt-f1-score": alt_f1})    
        wandb.finish()

if __name__=="__main__":
    ds = load_from_disk("hotpot").with_format("torch", device="cuda")
    CacheBlend().run(ds["hotpot_train"], ds["hotpot_valid"])

