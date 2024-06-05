import torch
from typing import List
from dataclasses import dataclass, field

class Config:
    # model 参数 ###########################
    # 情感分析模型，下载地址https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment
    Sentiment_model = "E:\\ai_model\\model\\Erlangshen-Roberta-330M-Sentiment"
    # 文本生成模型,下载地址 https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat
    gpt_model = "E:\\ai_model\\model\\qwen0.5"
    data_path = "data/train_data.json"
    save_lora_path = "E:\\ai_model\\model\\ppo\\save_lora"
    save_v_head_path = "E:\\ai_model\\model\\ppo\\v_head\\pytorch_model.bin"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    epochs = 10
    lr = 0.001
    # PPO 参数 ############################
    ppo_epochs = 3
    kl_ctl_value = 0.2
    gamma = 1.0  # 用于优势计算的折扣因子。控制未来奖励的重要性。
    lam = 0.95  # 用于优势计算的Lambda参数。它用于控制对未来奖励的考虑程度，结合时间差异方法。
    cliprange_value = 0.2  # 损失计算中值函数的裁剪范围。裁剪可以防止极端值对训练过程的负面影响。
    cliprange = 0.2  # PPO策略梯度损失中的裁剪范围。这个裁剪范围用于限制策略更新的步长，从而保持训练的稳定性。
    vf_coef = 0.1


@dataclass
class LoraArguments:
    lora_r: int = 2
    lora_alpha: int = 8
    lora_dropout: float = 0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['k_proj',  'v_proj']
    )
    # lora_target_modules = None
    lora_weight_path: str = ""
    q_lora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    is_reload_trained_params = True  # 是否接着上次训练模型继续训练
