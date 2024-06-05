import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from RLHF_PPO.config import Config
from RLHF_PPO.utils.tools import Tools
from peft import LoraConfig, get_peft_model, PeftModel
from RLHF_PPO.config import LoraArguments


class LoraModel(PeftModel):
    def __init__(self, config: Config, model):
        lora_args = LoraArguments()
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            task_type="CAUSAL_LM",
        )
        super().__init__(model, lora_config)
        self.v_head = torch.nn.Linear(1024, 1, bias=False).to(config.device)
        if lora_args.is_reload_trained_params:
            super().from_pretrained(model, config.save_lora_path)
            self.v_head.load_state_dict(torch.load(config.save_v_head_path))
        for name, module in self.named_modules():
            if 'lora_' in name:
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask, tools: Tools):
        res = super().forward(input_ids, attention_mask, output_hidden_states=True)
        values = self.v_head(res.hidden_states[0]).squeeze(-1)[:, :-1]
        values = tools.filter_mask(values)
        probs = tools.probs_from_logits(res.logits[:, :-1, :], input_ids[:, 1:])
        probs = tools.filter_mask(probs)
        return probs, values


class ActorCriticLoraModel(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(config.gpt_model).to(config.device).eval()
        self.model = LoraModel(config, model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.gpt_model)

    def forward(self, input_ids, attention_mask, tools: Tools):
        probs, values = self.model(input_ids, attention_mask, tools)
        return probs, values

    @torch.no_grad()
    def actor_generate(self, input_ids):
        generated_ids = self.model.generate(input_ids, max_new_tokens=512, top_p=1.0,
                                            num_beams=1,
                                            do_sample=False)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response_id = generated_ids[:, input_ids.shape[1]:]
        return response, generated_ids, response_id
