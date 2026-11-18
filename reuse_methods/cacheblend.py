from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache
import torch
import wandb
from helpers.hotpot_evaluate_v1 import f1_score
from datasets import Dataset, load_from_disk
import torch.nn.functional as F 
import reuse_methods.template as template

class CacheBlend(Template):
    def __init__(self, **kwargs):
        super().__init__()
        config.num_hidden_layers = 2
        self.trunc_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, dtype=model_dtype).to(self.device)

    def build_cache(self, cache: DynamicCache, docs: list[torch.Tensor], is_alt: bool):
        docs = valid_example["docs"]
        doc_pos = torch.tensor([], dtype=torch.long)
        p = cache.get_seq_length()
        offset = p
        for doc in docs: 
            doc_pos = torch.cat((doc_pos, offset + torch.arange(doc.shape[0])), dim=0)
            offset += doc.shape[0]
        doc_mask = torch.block_diag(*[torch.ones(d.shape[0], d.shape[0]).tril() for d in docs]).bool()[None, None]
        docs = torch.cat(docs)
        doc_cache = write_model(input_ids=docs, position_ids=doc_pos[None], attention_mask=doc_mask).past_key_values
        temp_cache = DynamicCache()
        temp_cache.layers = true_cache.layers[:2]
        true_cache = self.trunc_model(input_ids=docs[None], past_key_values=temp_cache).past_key_values
        
        dist_k, dist_v = [torch.linalg.vector_norm(getattr(true_cache.layers[1], x)[:, :, p:].float()-getattr(doc_cache.layers[1], x)[:, :, :].float(), dim=(0,1,3)) for x in ["keys", "values"]]
        score = dist_k / dist_k.mean() + dist_v / dist_v.mean()
        tok_pos = score.topk(int(r * (offset - p))).indices.sort().values
        tok_mask = torch.zeros((offset - p), dtype=torch.bool).index_fill(0, tok_pos, True)
        for l, layer in enumerate(doc_cache.layers):
            cache.update(layer.keys[:, :, ~tok_mask], layer.values[:, :, ~tok_mask], l)
        mask = ((p + tok_pos).unsqueeze(1) >= torch.cat((torch.arange(p), p + torch.arange(doc_len)[~tok_mask], p + tok_pos)))[None, None]
        if tok_mask.any(): self.model(input_ids=docs[tok_mask][None], attention_mask=mask, position_ids=(p + tok_pos)[None], past_key_values=cache, logits_to_keep=1)
        return cache
    


