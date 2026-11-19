from transformers import AutoModelForCausalLM, DynamicCache
import torch
from utils.template import Template
from cross_model_reuse.kv_translator import KVTranslator
import copy
from utils.hotpot_evaluate_v1 import f1_score as compute_f1

class MyMethod(Template):
    def __init__(self, kvt: KVTranslator, recompute_ratio: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        config = copy.deepcopy(self.model.config)
        config.num_hidden_layers = 2
        self.r = recompute_ratio
        self.trunc_model = AutoModelForCausalLM.from_pretrained(self.model.name_or_path, config=config, dtype=self.model.dtype).to(self.device)
        self.trunc_model.requires_grad_(False)
        self.trunc_model.eval()
        self.kvt = kvt


    def build_cache(self, cache: DynamicCache, docs: list[torch.Tensor], variant: str):
        doc_pos = torch.tensor([], dtype=torch.long, device=self.device)
        p = cache.get_seq_length()
        offset = p
        for doc in docs: 
            doc_pos = torch.cat((doc_pos, offset + torch.arange(doc.shape[0], device=self.device)), dim=0)
            offset += doc.shape[0]
        doc_mask = torch.block_diag(*[torch.ones((d.shape[0], d.shape[0]), device=self.device).tril() for d in docs]).bool()[None, None]
        docs = torch.cat(docs)
        write_model = self.alt_model if "alt" in variant else self.model

        doc_cache = write_model(input_ids=docs[None], position_ids=doc_pos[None], attention_mask=doc_mask).past_key_values
        if "tr" in variant: 
            tr_type = "base" if "base" in variant else "no_sink"
            doc_cache = self.kvt.translate(doc_cache, doc_pos, tr_type)
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




