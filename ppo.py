import torch
from config import Config
from utils.tools import Tools


class PPO:
    def __init__(self, actor_critic_model, config: Config, actor_critic_opt):
        self.actor_critic_model = actor_critic_model
        self.config = config
        self.actor_critic_opt = actor_critic_opt

    def train(self, prompt_generate_ids, attention_mask, prob_refs, reward, tools: Tools):
        with torch.no_grad():
            _, old_values = self.actor_critic_model(prompt_generate_ids, attention_mask, tools)  # 计算每个token的价值
        for _ in range(self.config.ppo_epochs):
            # 获得actor_critic模型新的probs和token对应的价值
            new_probs, new_values = self.actor_critic_model(prompt_generate_ids, attention_mask, tools)
            # 计算奖励值
            rewards, non_score_rewards = self.compute_rewards(reward, new_probs, prob_refs)  # 计算reward
            loss = self.loss(new_probs=new_probs, old_values=old_values, new_values=new_values,
                             rewards=rewards, old_probs=prob_refs)

            self.actor_critic_opt.zero_grad()
            loss.backward()
            self.actor_critic_opt.step()
            print(loss)

    def loss(self, new_probs, old_values, new_values, rewards, old_probs):
        """
        计算actor模型和评价模型的loss
        :param new_probs: actor模型生成的probs
        :param old_values: ppo 优化之前的价值
        :param new_values: ppo 优化过程中新的价值
        :param rewards: 每次生成token对应的奖励
        :param old_probs: reference模型生成的probs
        :return: actor loss 和 critic loss
        """
        """Calculate policy and value losses."""
        loss = torch.tensor(0.0)
        for new_prob, old_value, new_value, reward, old_prob in zip(new_probs, old_values, new_values, rewards,
                                                                    old_probs):
            new_prob = new_prob.unsqueeze(0)
            old_value = old_value.unsqueeze(0)
            new_value = new_value.unsqueeze(0)
            reward = reward.unsqueeze(0)
            old_prob = old_prob.unsqueeze(0)
            last_gae_lam = 0
            advantages_reversed = []
            gen_len = new_prob.shape[1]
            # GAE 计算优势函数，当前token获得的奖励(真实的) + 未来获得的价值(这个是上帝视角，不包含当前token) - 包含当前token在上帝视角下的价值
            # 当前token获得的奖励(真实的) + 未来获得的价值(这个是上帝视角，不包含当前token) 比 包含当前token在上帝视角下的价值 要准
            for t in reversed(range(gen_len)):
                next_values = old_value[:, t + 1] if t < gen_len - 1 else 0.0
                delta = reward[:, t] + self.config.gamma * next_values - old_value[:, t]
                last_gae_lam = delta + self.config.gamma * self.config.lam * last_gae_lam
                advantages_reversed.append(last_gae_lam)
            advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
            returns = advantages + old_value  # Q值，当前token获得的奖励(真实的) + 未来获得的价值(这个是上帝视角，不包含当前token)
            advantages = self.whiten(advantages)
            advantages = advantages.detach()
            value_clipped = torch.clamp(new_value,
                                        old_value - self.config.cliprange_value,
                                        old_value + self.config.cliprange_value)  # 截断防止训练废了
            vf_loss1 = (new_value - returns) ** 2  # 上帝视角的价值减去Q值的误差，用于优化上帝模型
            vf_loss2 = (value_clipped - returns) ** 2
            vf_loss = torch.mean(torch.max(vf_loss2, vf_loss1))

            ratio = torch.exp(new_prob - old_prob)  # 控制优化范围，防止训练离原始模型偏差过大
            pg_losses = -advantages * ratio  # importance sampling
            pg_losses2 = -advantages * torch.clamp(ratio,
                                                   1.0 - self.config.cliprange,
                                                   1.0 + self.config.cliprange)  # 截断防止训练废了
            pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
            loss += pg_loss + self.config.vf_coef * vf_loss
        return loss

    def compute_rewards(self, scores, probs, ref_probs):
        """
        计算reward值,由于对每一个token不能给与即使的奖励，这里使用kl散度补偿
        :param scores:reward model给出的奖励值，每条句子只有一个值
        :param probs: actor model生成的probs
        :param ref_probs: reference model 生成的probs
        :return: 返回每个token的奖励值
        """
        rewards, non_score_rewards = [], []
        for score, prob, ref_prob in zip(scores, probs, ref_probs):
            kl = prob - ref_prob  # (seq_len, )
            non_score_reward = -self.config.kl_ctl_value * kl  # (seq_len, )
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()  # 前面每一个token的reward都来自KL惩罚
            reward[-1] += score  # 在最后一位加上人工给的reward
            rewards.append(reward)
        return rewards, non_score_rewards  # (batch, seq_len)

    @staticmethod
    def whiten(values, shift_mean=True):
        """
        归一化
        :param values: 要归一化的值
        :param shift_mean: 负一化方式
        :return: 返回归一化之后的结果
        """
        mean, var = torch.mean(values), torch.var(values)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened
