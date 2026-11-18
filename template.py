from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch
import wandb
from datasets import Dataset
from hotpot_evaluate_v1 import f1_score as compute_f1
from collections.abc import Callable


class Template():
    def __init__(self, model: AutoModelForCausalLM, 
                 alt_model: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer, 
                 device: torch.device | str = "cuda", 
                 max_new_tokens: int=64):
        self.device = device
        self.model = model.to(device)
        self.alt_model = alt_model.to(device)
        self.alt_model.eval()
        self.model.config.use_cache = True
        self.alt_model.config.use_cache = True
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def build_doc_cache(self, prefix_cache: DynamicCache, docs: list[torch.Tensor], is_alt: bool):
        return

    def train(self, train_dataset: Dataset):
        return
    
    def validate(self, valid_dataset: Dataset, is_alt: bool = False):
        with torch.no_grad():
            f1 = 0
            for valid_example in valid_dataset:
                prefix_cache = self.model(input_ids=valid_example["prefix"][None]).past_key_values
                cache = self.build_doc_cache(prefix_cache, valid_example["docs"], is_alt)
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
            f1 = f1 / len(valid_dataset)
        return f1
 
    def run(self, valid_dataset: Dataset, train_dataset: Dataset):
        wandb.init()
        self.model.train()
        self.train(train_dataset: Dataset)
        self.model.eval()
        f1 = self.validate(valid_dataset, self.build_doc_cache)
        alt_f1 = self.validate(valid_dataset, self.build_doc_cache, is_alt=True)
        wandb.log({"f1": f1, "alt_f1": alt_f1})
        wandb.finish()


