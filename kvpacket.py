

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
import wandb
import torch.nn.functional as F
from datasets import Dataset
from hotpot_evaluate_v1 import f1_score
from vllm import LLM, SamplingParams
from datasets import load_from_disk 

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

class KVPacket():
    def __init__(self, model_path: str = "Qwen2.5-1.5B-Instruct", 
                 alt_model_path: str = "Qwen2.5-Coder-1.5B-Instruct", 
                 tokenizer_path: str = "Qwen2.5-1.5B-Instruct", 
                 model_dtype: torch.dtype = torch.bfloat16, 
                 device: torch.device | str = "cuda", 
                 header_len: int = 8, 
                 lr: float=1e-3, 
                 max_new_tokens: int=64):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=model_dtype).to(self.device)
        self.alt_model = AutoModelForCausalLM.from_pretrained(alt_model_path, dtype=model_dtype).to(self.device)
        self.model.config.use_cache = True
        self.alt_model.config.use_cache = True
        self.max_new_tokens = max_new_tokens
        self.model.requires_grad_(False)
        self.packetize = Packetize(self.device, self.model.config.hidden_size, header_len)
        self.optimizer = torch.optim.AdamW(self.packetize.parameters(), lr=lr)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.vllm = LLM(model_path, dtype=model_dtype, gpu_memory_utilization=0.3, enable_sleep_mode=True)
       

    def validate(self, valid_examples: Dataset, is_alt: bool = False):
        write_model = self.alt_model if is_alt else self.model
        write_embed = write_model.get_input_embeddings()
        self.model.eval()  
        self.alt_model.eval()  
        with torch.no_grad():
            score = 0
            for valid_example in valid_examples:
                packetized_docs = [self.packetize(write_embed(doc)) for doc in valid_example["docs"]]
                doc_mask = torch.block_diag(*[torch.ones((pd.shape[0], pd.shape[0]), dtype=torch.bool, device=self.device).tril() for pd in packetized_docs])[None, None]
                pos_ids = valid_example["prefix"].shape[0] + torch.arange(doc_mask.shape[-1], device=self.device)
                doc_cache = write_model(inputs_embeds = torch.cat(packetized_docs)[None], attention_mask=doc_mask, position_ids=pos_ids[None]).past_key_values
                cache = self.model(input_ids = valid_example["prefix"][None]).past_key_values
                for i, layer in enumerate(doc_cache.layers): cache.update(layer.keys, layer.values, i)
                output_tokens = []
                output = self.model(input_ids = valid_example["suffix"][None], past_key_values=cache, logits_to_keep=1)
                for _ in range(self.max_new_tokens):
                    output_tokens.append(output.logits[0, 0].argmax())
                    if output_tokens[-1]==self.tokenizer.eos_token_id:
                        break
                    output = self.model(input_ids = output_tokens[-1].reshape(1,1), past_key_values=output.past_key_values, logits_to_keep=1)
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                answer_text = self.tokenizer.decode(valid_example["answer"], skip_special_tokens=True)
                score += f1_score(output_text, answer_text)[0] 

            score = score / len(valid_examples)
        self.model.train()
        return score
 
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, step_size: int=32, valid_freq:float=0.01, valid_size: int=100):
        wandb.init()
        self.model.train()
        valid_examples = valid_dataset.select(range(valid_size))
        t_output = self.vllm.generate([{"prompt_token_ids": torch.cat([example["prefix"], *example["docs"], example["suffix"]])} for example in train_dataset.with_format(None)], SamplingParams(temperature=0, max_tokens=self.max_new_tokens))
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
            
            loss = F.kl_div(F.log_softmax(s_output_logits, dim=1), F.log_softmax(t_output_logits, dim=1), log_target=True, reduction="batchmean")
            wandb.log({"train-loss": loss.item()})
            (loss / step_size).backward()
            if step % step_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            if step % int(valid_freq * len(train_dataset)) == 0:
                f1 = self.validate(valid_examples)
                wandb.log({"f1-score": f1})
                alt_f1 = self.validate(valid_examples, is_alt=True)
                wandb.log({"alt-f1-score": alt_f1})
                
        wandb.finish()

if __name__=="__main__":
    ds = load_from_disk("hotpot").with_format("torch", device="cuda")
    KVPacket.train(ds["hotpot_train"], ds["hotpot_valid"])