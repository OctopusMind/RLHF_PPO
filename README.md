# RLHF PPO
## 博客地址：
## 项目描述：
本仓库实现PPO算法，由于个人硬件有限强化的模型是qwen_0.5B, 使用lora调节模型参数。
奖励模型使用的是Erlangshen-Roberta-330M-Sentiment，不需要微调这个模型，下载地址：https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment
## 代码组织解释
### 训练数据
data/train_data.json 此数据是自己造的,仅用于学习使用。虽然数据就2条，我微调了20个epochs，效果还行。后面会附带训练后的结果
### model 文件
model/actor_critic_model.py 这里actor和critic模型使用同一个底座，没有拆开
model/reference_model.py 参考模型，其实就是原始的qwen模型
model/reward_model.py 奖励模型，使用的是Erlangshen-Roberta-330M-Sentiment
### utils 文件
这里主要有两个文件，utils/data_load.py加载数据和utils/tools.py模型这几个模型都需要调用的功能独立出来
### config.py 配置文件
### ppo.py PPO核心实现
### main.py 训练代码
### inference.py 训练完成之后，使用该文件预测效果


## 微调后效果比对
