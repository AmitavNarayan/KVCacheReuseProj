from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
import torch.nn.functional as F
from datasets import Dataset
from vllm import LLM, SamplingParams
from datasets import load_from_disk 
from reuse_methods.template import Template



class Packetize(torch.nn.Module):
    def __init__(self, device: str | torch.device, d_model: int, header_len: int, mean: float = 0, stdev: float = 0.02):
        super().__init__()
        self.header = torch.nn.Parameter(torch.zeros((header_len, d_model), device=device))
        self.trailer = torch.nn.Parameter(torch.zeros((header_len, d_model), device=device))
        torch.nn.init.normal_(self.header, mean=mean, std=stdev)
        torch.nn.init.normal_(self.trailer, mean=mean, std=stdev)
    def forward(self, x: torch.Tensor):
        new_x = torch.cat((self.header.to(x.dtype), x, self.trailer.to(x.dtype)), dim=-2)
        return new_x 

class KVPacket(Template):
    def __init__(self, model_dtype: torch.dtype = torch.bfloat16, 
                 header_len: int = 8, 
                 lr: float=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.model.requires_grad_(False)
        self.packetize = Packetize(self.device, self.model.config.hidden_size, header_len)
        self.optimizer = torch.optim.AdamW(self.packetize.parameters(), lr=lr)
        self.vllm = LLM(self.model.config.name_or_path, dtype=model_dtype, gpu_memory_utilization=0.3, enable_sleep_mode=True)

    def build_cache(self, cache: DynamicCache, docs: list[torch.Tensor], is_alt: bool):
        write_model = self.alt_model if is_alt else self.model
        write_embed = write_model.get_input_embeddings()
        packets = [self.packetize(write_embed(doc)) for doc in docs]
        doc_mask = torch.block_diag(*[torch.ones((pd.shape[0], pd.shape[0]), dtype=torch.bool, device=self.device).tril() for pd in packets])[None, None]
        pos_ids = cache.get_seq_length() + torch.arange(doc_mask.shape[-1], device=self.device)
        doc_cache = write_model(inputs_embeds = torch.cat(packets)[None], attention_mask=doc_mask, position_ids=pos_ids[None]).past_key_values 
        for l, layer in enumerate(doc_cache.layers):
            cache.update(layer.keys, layer.values, l)
        return cache
        
    def train(self, train_dataset: Dataset, step_size: int):
        t_output = self.vllm.generate([{"prompt_token_ids": torch.cat([example["prefix"], *example["docs"], example["suffix"]]).tolist()} for example in train_dataset], SamplingParams(temperature=0, max_tokens=self.max_new_tokens))
        self.vllm.sleep(level=2)
        t_output_tokens = [torch.tensor(x.outputs[0].token_ids, device=self.device) for x in t_output]
        for step, example in enumerate(train_dataset, start=1):
            embed = self.model.get_input_embeddings()
            answer = t_output_tokens[step-1]
            t_tokens = torch.cat((example["prefix"], *example["docs"], example["suffix"], answer))
            with torch.no_grad(): t_output_logits = self.model(input_ids=t_tokens[None], logits_to_keep=len(answer)+1).logits[0, :-1]
            t_output_embeds = embed(answer)

            segments = [embed(example["prefix"]), *[self.packetize(embed(doc)) for doc in example["docs"]], torch.cat((embed(example["suffix"]), t_output_embeds))]
            mask = torch.block_diag(*[torch.ones((s.shape[0], s.shape[0]), device=self.device, dtype=torch.bool).tril() for s in segments])
            mask[-segments[-1].shape[0]:, :-segments[-1].shape[0]] = True
            s_output_logits = self.model(inputs_embeds=torch.cat(segments)[None], attention_mask=mask[None, None], logits_to_keep=len(answer)+1).logits[0, :-1]
            
            loss = F.kl_div(F.log_softmax(s_output_logits.float(), dim=1), F.log_softmax(t_output_logits.float(), dim=1), log_target=True, reduction="batchmean")
            (loss / step_size).backward()
            if step % step_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                





