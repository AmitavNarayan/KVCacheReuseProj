from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
import wandb
from datasets import Dataset
from utils.hotpot_evaluate_v1 import f1_score as compute_f1


class Template():
    def __init__(self, model: AutoModelForCausalLM, 
                 alt_model: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer, 
                 forward_dtype: torch.dtype = torch.bfloat16,
                 device: torch.device | str = "cuda", 
                 max_new_tokens: int=64,
                 variants = ("main", "alt", "alt_tr_base", "alt_tr_no_sink")):
        self.device = device
        self.model = model.to(device)
        self.alt_model = alt_model.to(device)
        self.alt_model.eval()
        self.alt_model.requires_grad_(False)
        self.model.config.use_cache = True
        self.alt_model.config.use_cache = True
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.forward_dtype = forward_dtype
        self.variants = variants

    def build_cache(self, prefix_cache: DynamicCache, docs: list[torch.Tensor], variant: str):
        return
    
    def train(self, train_dataset: Dataset, step_size: int):
        return
    
    def validate(self, valid_dataset: Dataset):
        variant_f1s = dict([(v, None) for v in self.variants])
        for variant in self.variants:
            with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.forward_dtype):
                f1 = 0
                for valid_example in valid_dataset:
                    prefix_cache = self.model(input_ids=valid_example["prefix"][None]).past_key_values
                    cache = self.build_cache(prefix_cache, valid_example["docs"], variant)
                    output_tokens = []
                    output = self.model(input_ids = valid_example["suffix"][None], past_key_values=cache, logits_to_keep=1)
                    for _ in range(self.max_new_tokens):
                        output_tokens.append(output.logits[0, 0].argmax())
                        if output_tokens[-1]==self.tokenizer.eos_token_id:
                            break
                        output = self.model(input_ids = output_tokens[-1].reshape(1,1), past_key_values=output.past_key_values, logits_to_keep=1)
                    output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                    answer_text = self.tokenizer.decode(valid_example["answer"], skip_special_tokens=True)
                    f1 += compute_f1(output_text, answer_text)[0] 
                variant_f1s[variant] = f1 / len(valid_dataset)
        return variant_f1s
 
    def run(self, train_dataset: Dataset, valid_dataset: Dataset, step_size: int = 32):
        wandb.init()
        self.model.train()
        self.train(train_dataset, step_size)
        self.model.eval()
        f1_scores = self.validate(valid_dataset)
        wandb.log(f1_scores)
        wandb.finish()


