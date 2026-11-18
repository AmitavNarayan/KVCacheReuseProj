

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
import wandb
import torch.nn.functional as F
from datasets import Dataset
from vllm import LLM, SamplingParams
from datasets import load_from_disk 
from template import Template



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
    def __init__(self,
                 model_dtype: torch.dtype = torch.bfloat16, 
                 header_len: int = 8, 
                 lr: float=1e-3, 
                 max_new_tokens: int=64,
                 **kwargs):
        super().__init__(**kwargs)
        self.packetize = Packetize(self.device, self.model.config.hidden_size, header_len)
        self.optimizer = torch.optim.AdamW(self.packetize.parameters(), lr=lr)
        self.vllm = LLM(model_path, dtype=model_dtype, gpu_memory_utilization=0.3, enable_sleep_mode=True)

    def build_doc_cache(self, pc, docs):
        for i, doc in enumerate(docs)
            doc[i] = self.
        
    def train(self, train_dataset):
        t_output = self.vllm.generate([{"prompt_token_ids": torch.cat([example["prefix"], *example["docs"], example["suffix"]]).tolist()} for example in train_dataset], SamplingParams(temperature=0, max_tokens=self.max_new_tokens))
        self.vllm.sleep(level=2)
        t_output_tokens = [torch.tensor(x.outputs[0].token_ids, device=self.device) for x in t_output]
        for step, example in enumerate(train_dataset, start=1):
            embed = self.model.get_input_embeddings()
            answer = t_output_tokens[step-1]
            t_tokens = torch.cat((example["prefix"], *example["docs"], example["suffix"], answer))
            t_output_logits = self.model(input_ids=t_tokens[None], logits_to_keep=len(answer)+1).logits[0, :-1]
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
                

if __name__=="__main__":
    ds = load_from_disk("hotpot").with_format("torch", device="cuda")
    KVPacket().run(ds["hotpot_train"], ds["hotpot_valid"])





