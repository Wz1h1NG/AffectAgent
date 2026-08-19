"""Token-level MAPPO update used by the official AffectAgent implementation."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute equation (9) for a single terminal-ended token trajectory."""

    rewards = rewards.reshape(-1)
    values = values.reshape(-1)
    if rewards.numel() != values.numel():
        raise ValueError("rewards and values must contain the same number of time steps")
    if rewards.numel() == 0:
        return rewards.clone(), values.clone()
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros((), dtype=values.dtype, device=values.device)
    next_value = torch.zeros((), dtype=values.dtype, device=values.device)
    for step in range(rewards.numel() - 1, -1, -1):
        delta = rewards[step] + gamma * next_value - values[step]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[step] = next_advantage
        next_value = values[step]
    returns = advantages + values
    return advantages, returns


class SequenceValueHead(nn.Module):
    """Critic applied independently to every generated-token hidden state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.summary = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)
        hidden_states = hidden_states.to(dtype=self.summary[0].weight.dtype)
        return self.summary(hidden_states).squeeze(-1)


class PolicyGradientUpdater:
    """Joint Actor-Critic PPO with terminal task/KL reward and token-level GAE."""

    def __init__(
        self,
        model,
        reference_model,
        value_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        kl_coef: float = 0.1,
        ppo_epochs: int = 4,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.reference_model = reference_model
        self.value_head = value_head
        self.optimizer = optimizer
        self.kl_coef = kl_coef
        self.ppo_epochs = ppo_epochs
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.device = device

    @staticmethod
    def _aligned_length(*tensors: torch.Tensor) -> int:
        return min(tensor.reshape(-1).numel() for tensor in tensors)

    def _build_replay_buffer(self, trajectories, pipeline) -> List[Dict[str, object]]:
        replay_buffer = []
        self.reference_model.eval()
        for trajectory in trajectories:
            result = trajectory["result"]
            agent_name = str(trajectory["agent"])
            task_reward = float(trajectory["reward"])
            with torch.no_grad():
                old_log_probs, old_states = pipeline.compute_agent_log_probs_and_state(
                    result,
                    agent_name,
                    self.model.llama_model,
                )
                ref_log_probs = pipeline.compute_agent_log_probs(
                    result,
                    agent_name,
                    self.reference_model,
                )
                if old_states is None:
                    continue
                old_values = self.value_head(old_states)

            if old_log_probs.numel() == 0:
                continue
            length = self._aligned_length(old_log_probs, ref_log_probs, old_values)
            old_log_probs = old_log_probs[:length]
            ref_log_probs = ref_log_probs[:length]
            old_values = old_values[:length]

            # Equation (10): supervision and sequence KL are terminal-only.
            sequence_kl = (old_log_probs - ref_log_probs).sum()
            terminal_reward = torch.as_tensor(
                task_reward,
                dtype=old_values.dtype,
                device=old_values.device,
            ) - self.kl_coef * sequence_kl
            rewards = torch.zeros_like(old_values)
            rewards[-1] = terminal_reward
            advantages, returns = compute_gae(
                rewards,
                old_values,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            replay_buffer.append({
                "result": result,
                "agent": agent_name,
                "task_reward": task_reward,
                "regularized_reward": float(terminal_reward),
                "old_log_probs": old_log_probs.detach(),
                "ref_log_probs": ref_log_probs.detach(),
                "old_values": old_values.detach(),
                "returns": returns.detach(),
                "advantages": advantages.detach(),
            })

        if replay_buffer:
            flat = torch.cat([item["advantages"] for item in replay_buffer])
            mean = flat.mean()
            std = flat.std(unbiased=False)
            if flat.numel() > 1 and float(std) > 1e-8:
                for item in replay_buffer:
                    item["advantages"] = ((item["advantages"] - mean) / std).detach()
        return replay_buffer

    def update_step(self, trajectories, pipeline) -> Dict[str, float]:
        if not trajectories:
            return self._empty_stats()

        self.model.llama_model.eval()
        self.value_head.eval()
        replay_buffer = self._build_replay_buffer(trajectories, pipeline)
        if not replay_buffer:
            return self._empty_stats()

        self.model.llama_model.train()
        self.value_head.train()
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "kl": 0.0, "count": 0}
        update_rounds = 0

        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            epoch_items = 0
            for item in replay_buffer:
                policy_log_probs, states = pipeline.compute_agent_log_probs_and_state(
                    item["result"],
                    item["agent"],
                    self.model.llama_model,
                )
                if policy_log_probs.numel() == 0 or states is None:
                    continue
                value_predictions = self.value_head(states)
                length = self._aligned_length(
                    policy_log_probs,
                    value_predictions,
                    item["old_log_probs"],
                    item["old_values"],
                    item["returns"],
                    item["advantages"],
                )
                policy_log_probs = policy_log_probs[:length]
                value_predictions = value_predictions[:length]
                old_log_probs = item["old_log_probs"][:length]
                advantages = item["advantages"][:length]

                ratio = torch.exp(torch.clamp(policy_log_probs - old_log_probs, -20.0, 20.0))
                surrogate_1 = ratio * advantages
                surrogate_2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                ) * advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                old_values = item["old_values"][:length]
                value_targets = item["returns"][:length]
                clipped_values = old_values + torch.clamp(
                    value_predictions - old_values,
                    -self.clip_range,
                    self.clip_range,
                )
                value_loss = 0.5 * torch.max(
                    (value_predictions - value_targets).pow(2),
                    (clipped_values - value_targets).pow(2),
                ).mean()
                # KL is already included in the terminal reward; this is diagnostic only.
                kl = (policy_log_probs - item["ref_log_probs"][:length]).mean()
                loss = (policy_loss + self.value_coef * value_loss) / max(len(replay_buffer), 1)
                loss.backward()

                totals["loss"] += float(loss.detach()) * max(len(replay_buffer), 1)
                totals["policy"] += float(policy_loss.detach())
                totals["value"] += float(value_loss.detach())
                totals["kl"] += float(kl.detach())
                totals["count"] += 1
                epoch_items += 1

            if epoch_items:
                torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
                self.optimizer.step()
                update_rounds += 1

        if not totals["count"]:
            return self._empty_stats()
        count = totals["count"]
        return {
            "loss": totals["loss"] / count,
            "mean_reward": sum(item["task_reward"] for item in replay_buffer) / len(replay_buffer),
            "mean_regularized_reward": sum(item["regularized_reward"] for item in replay_buffer) / len(replay_buffer),
            "mean_kl": totals["kl"] / count,
            "policy_loss": totals["policy"] / count,
            "value_loss": totals["value"] / count,
            "update_rounds": update_rounds,
        }

    @staticmethod
    def _empty_stats() -> Dict[str, float]:
        return {
            "loss": 0.0,
            "mean_reward": 0.0,
            "mean_regularized_reward": 0.0,
            "mean_kl": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "update_rounds": 0,
        }
