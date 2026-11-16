from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache
import torch
import wandb
from hotpot_evaluate_v1 import f1_score
from datasets import Dataset, load_from_disk
from cross_model.translator import Translate
import argparse


class MyMethod():
    def __init__(self, max_new_tokens: int = 64, 
                 compute_dtype=torch.bfloat16, 
                 model_path: str = "/scratch/an4245/Qwen2.5-7B-Instruct", 
                 alt_model_path: str = "/scratch/an4245/Qwen2.5-Coder-7B-Instruct", 
                 tokenizer_path: str = "/scratch/an4245/Qwen2.5-Coder-7B-Instruct", 
                 model_dtype: torch.dtype = torch.bfloat16, 
                 alt_model_dtype: torch.dtype = torch.bfloat16,
                 device: torch.device | str = "cuda",):
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
        self.translate = Translate()

    @staticmethod 
    def cache_to_kv(cache: DynamicCache):
        ret = [torch.stack([getattr(l, x) for l in cache.layers]).squeeze(1) for x in ["keys", "values"]]
        return ret
    def validate(self, r: float, valid_examples: Dataset, is_alt: bool = False):
        write_model = self.alt_model if is_alt else self.model
        self.model.eval()
        f1 = 0
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.compute_dtype):
            for valid_example in valid_examples:
                p = valid_example["prefix"].shape[0]
                docs = valid_example["docs"]
                doc_len = sum(d.shape[0] for d in docs)
                doc_pos = torch.arange(p, p + doc_len, device=self.device)
                prefix_docs = torch.cat((valid_example["prefix"], *docs))
                doc_mask = torch.block_diag(*[torch.ones((d.shape[0], d.shape[0]), device=self.device).tril() for d in docs]).bool()[None, None]
                doc_kv = MyMethod.cache_to_kv(write_model(input_ids=prefix_docs[p:][None], position_ids=doc_pos[None], attention_mask=doc_mask).past_key_values)
                if is_alt: doc_kv = self.translate.translate(doc_kv, doc_pos)
                cache = self.model(input_ids=prefix_docs[:p][None]).past_key_values
                true_kv = MyMethod.cache_to_kv(self.trunc_model(input_ids=prefix_docs[None]).past_key_values)
                dist_k, dist_v = [torch.linalg.vector_norm(true_kv[i][1][:, p:].float()-doc_kv[i][1].float(), dim=(0,2)) for i in range(2)]
                score = dist_k / dist_k.mean() + dist_v / dist_v.mean()
                tok_pos = score.topk(int(r * doc_len)).indices.sort().values
                tok_mask = torch.zeros(doc_len, dtype=torch.bool, device=self.device).index_fill(0, tok_pos, True)
                for l in range(self.layers): cache.update(doc_kv[0][l][:, ~tok_mask][None], doc_kv[1][l][:, ~tok_mask][None], l)
                mask = ((p + tok_pos).unsqueeze(1) >= torch.cat((torch.arange(p, device=self.device), p + torch.arange(doc_len, device=self.device)[~tok_mask], p + tok_pos)))[None, None]
                if tok_mask.any(): self.model(input_ids=prefix_docs[p:][tok_mask][None], attention_mask=mask, position_ids=(p + tok_pos)[None], past_key_values=cache, logits_to_keep=1)
                output_tokens = []
                to_add = valid_example["suffix"][None]
                to_add_pos = (torch.arange(to_add.shape[1], device=self.device) + doc_len + p)[None]
                for _ in range(self.max_new_tokens):
                    output_tokens.append(self.model(input_ids = to_add, past_key_values=cache, position_ids=to_add_pos, logits_to_keep=1).logits[0,-1].argmax())
                    if output_tokens[-1] == self.tokenizer.eos_token_id:
                        break
                    to_add, to_add_pos = output_tokens[-1][None, None], to_add_pos[:, -1:]+1

                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                answer_text = self.tokenizer.decode(valid_example["answer"], skip_special_tokens=True)
                f1 += f1_score(output_text, answer_text)[0]
            return f1 / len(valid_examples)
    
    def run(self, valid_dataset: Dataset, valid_size: int=100, r: float = 0.15):
        wandb.init()
        valid_examples = valid_dataset.shuffle().select(range(valid_size))
        f1 = self.validate(r, valid_examples)
        alt_f1 = self.validate(r, valid_examples, is_alt=True)
        wandb.log({"f1-score": f1})
        wandb.log({"alt-f1-score": alt_f1})
        wandb.finish()

if __name__=="__main__":
    ds = load_from_disk("/scratch/an4245/hotpot").with_format("torch", device="cuda")
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=float, default=0.15)
    parser.add_argument("--valid_size", type=int, default=400)
    args = parser.parse_args()
    MyMethod().run(ds["hotpot_valid"], 
                r = args.r, valid_size = args.valid_size)
    ''' MyMethod(model_path="/scratch/an4245/Qwen2.5-O.5B-Instruct", 
             alt_model_path="/scratch/an4245/Qwen2.5-7B-Instruct", 
             tokenizer_path="/scratch/an4245/Qwen2.5-7B-Instruct").run(ds["hotpot_valid"], 
            r = input_r, valid_size = input_valid_size) '''
