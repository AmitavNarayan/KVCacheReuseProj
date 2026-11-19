from transformers import AutoModelForCausalLM, DynamicCache
import torch
from utils.template import Template
from cross_model_reuse.translator import Translate 
import copy
from datasets import Dataset
from utils.hotpot_evaluate_v1 import f1_score as compute_f1

class MyMethod(Template):
    def __init__(self, translator: Translate, recompute_ratio: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        config = copy.deepcopy(self.model.config)
        config.num_hidden_layers = 2
        self.r = recompute_ratio
        self.trunc_model = AutoModelForCausalLM.from_pretrained(self.model.name_or_path, config=config, dtype=self.model.dtype).to(self.device)
        self.trunc_model.requires_grad_(False)
        self.trunc_model.eval()
        self.translator = translator


    def build_cache(self, cache: DynamicCache, docs: list[torch.Tensor], is_alt: bool):
        write_model = self.alt_model if is_alt else self.model
        doc_pos = torch.tensor([], dtype=torch.long, device=self.device)
        p = cache.get_seq_length()
        offset = p
        for doc in docs: 
            doc_pos = torch.cat((doc_pos, offset + torch.arange(doc.shape[0], device=self.device)), dim=0)
            offset += doc.shape[0]
        doc_mask = torch.block_diag(*[torch.ones((d.shape[0], d.shape[0]), device=self.device).tril() for d in docs]).bool()[None, None]
        docs = torch.cat(docs)
        doc_cache = write_model(input_ids=docs[None], position_ids=doc_pos[None], attention_mask=doc_mask).past_key_values
        if is_alt: doc_cache = self.translator.translate(doc_cache, doc_pos)
        temp_cache = DynamicCache()
        for l in range(2): temp_cache.update(cache.layers[l].keys, cache.layers[l].values, l)
        true_cache = self.trunc_model(input_ids=docs[None], past_key_values=temp_cache).past_key_values
        
        dist_k, dist_v = [torch.linalg.vector_norm(getattr(true_cache.layers[1], x)[:, :, p:].float()-getattr(doc_cache.layers[1], x)[:, :, :].float(), dim=(0,1,3)) for x in ["keys", "values"]]
        score = dist_k / dist_k.mean() + dist_v / dist_v.mean()
        tok_pos = score.topk(int(self.r * (offset - p))).indices.sort().values
        tok_mask = torch.zeros((offset - p), device=self.device, dtype=torch.bool).index_fill(0, tok_pos, True)
        for l, layer in enumerate(doc_cache.layers):
            cache.update(layer.keys[:, :, ~tok_mask], layer.values[:, :, ~tok_mask], l)
        mask = ((p + tok_pos).unsqueeze(1) >= torch.cat((torch.arange(p, device=self.device), p + torch.arange(offset - p, device=self.device)[~tok_mask], p + tok_pos)))[None, None]
        if tok_mask.any(): self.model(input_ids=docs[tok_mask][None], attention_mask=mask, position_ids=(p + tok_pos)[None], past_key_values=cache, logits_to_keep=1)
        return cache

    def validate(self, valid_dataset: Dataset, is_alt: bool = False):
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.forward_dtype):
            f1 = 0
            for valid_example in valid_dataset:
                prefix_cache = self.model(input_ids=valid_example["prefix"][None]).past_key_values
                cache = self.build_cache(prefix_cache, valid_example["docs"], is_alt)
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



