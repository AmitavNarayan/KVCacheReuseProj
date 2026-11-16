from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache
import torch
import wandb
from hotpot_evaluate_v1 import f1_score
from datasets import Dataset, load_from_disk
from cross_model.translator import Translate
import sys


class MyMethod():
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
        self.translate = Translate()

    @staticmethod 
    def cache_to_kv(cache: DynamicCache):
        ret = [torch.stack([getattr(l, x) for l in cache.layers]).squeeze(1) for x in ["keys", "values"]]
        return ret
    def validate(self, r: float, valid_examples: Dataset, is_alt: bool = False):
        write_model = self.alt_model if is_alt else self.model
        self.model.eval()
        f1 = 0
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.compute_dtype), torch.device(self.device):
            for valid_example in valid_examples:
                p = valid_example["prefix"].shape[0]
                docs = valid_example["docs"]
                doc_pos = torch.tensor([], dtype=torch.long)
                offset = p
                prefix_docs = torch.cat((valid_example["prefix"], *docs))
                for doc in docs: 
                    doc_pos = torch.cat((doc_pos, offset + torch.arange(doc.shape[0])), dim=0)
                    offset += doc.shape[0]
                doc_len = offset - p
                doc_mask = torch.block_diag(*[torch.ones(d.shape[0], d.shape[0]).tril() for d in docs]).bool()[None, None]
                doc_kv = MyMethod.cache_to_kv(write_model(input_ids=prefix_docs[p:][None], position_ids=doc_pos[None], attention_mask=doc_mask).past_key_values)
                if is_alt: doc_kv = self.translate.translate(doc_kv, doc_pos)
                cache = self.model(input_ids=prefix_docs[:p][None]).past_key_values
                true_kv = MyMethod.cache_to_kv(self.trunc_model(input_ids=prefix_docs[None]).past_key_values)
                dist_k, dist_v = [torch.linalg.vector_norm(true_kv[i][1][:, p:].float()-doc_kv[i][1].float(), dim=(0,2)) for i in range(2)]
                score = dist_k / dist_k.mean() + dist_v / dist_v.mean()
                tok_pos = score.topk(int(r * doc_len)).indices.sort().values
                tok_mask = torch.zeros(doc_len, dtype=torch.bool).index_fill(0, tok_pos, True)
                for l in range(self.layers): cache.update(doc_kv[0][l][:, ~tok_mask][None], doc_kv[1][l][:, ~tok_mask][None], l)
                mask = ((p + tok_pos).unsqueeze(1) >= torch.cat((torch.arange(p), p + torch.arange(doc_len)[~tok_mask], p + tok_pos)))[None, None]
                if tok_mask.any(): self.model(input_ids=prefix_docs[p:][tok_mask][None], attention_mask=mask, position_ids=(p + tok_pos)[None], past_key_values=cache, logits_to_keep=1)
                output_tokens = []
                to_add = valid_example["suffix"][None]
                to_add_pos = (torch.arange(to_add.shape[1]) + doc_len + p)[None]
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
    ds = load_from_disk("hotpot").with_format("torch", device="cuda")
    input_r = float(sys.argv[1])
    input_valid_size = int(sys.argv[2])
    MyMethod().run(ds["hotpot_valid"], r = input_r, valid_size = input_valid_size)
