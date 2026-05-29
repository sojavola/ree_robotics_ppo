"""
ActorCritic CNN for PPO — ree_exploration_ppo.

Shared CNN backbone (identical to DQN QNetwork, Caccavale et al. 2023):

    Conv2d(C → 32,  8×8, stride=4)  ReLU  →  (B, 32, 24, 24)
    Conv2d(32 → 64, 4×4, stride=2)  ReLU  →  (B, 64, 11, 11)
    Conv2d(64 → 64, 3×3, stride=1)  ReLU  →  (B, 64,  9,  9)
    Flatten                                →  (B, 5184)
    Linear(5184, 512)                ReLU  →  (B, 512)

Actor head:  Linear(512, num_actions) → Categorical distribution
Critic head: Linear(512, 1)           → scalar V(s)
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """Shared-backbone Actor-Critic for PPO on global 100×100 REE maps.

    forward(obs)              → (action_probs, value)
    get_action(obs)           → (action, log_prob, value)   [agent inference]
    evaluate(obs, actions)    → (log_probs, values, entropy) [trainer update]
    """

    def __init__(self, obs_channels: int = 6, num_actions: int = 8) -> None:
        super().__init__()
        self._num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        flat = self._get_flat_size(obs_channels)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512),
            nn.ReLU(inplace=True),
        )

        self.actor_head = nn.Linear(512, num_actions)
        self.critic_head = nn.Linear(512, 1)

        # Orthogonal init (CleanRL / Stable-Baselines3 standard for PPO):
        # - Conv + fc layers: gain=sqrt(2) for ReLU activations
        # - Actor head: gain=0.01 → near-uniform initial policy (max entropy start)
        # - Critic head: gain=1.0  → reasonable initial value estimates
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init — standard PPO practice (CleanRL, SB3)."""
        for module in self.conv.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Small gain for actor → near-uniform policy at init → high entropy start
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        # Standard gain for critic
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    def _get_flat_size(self, obs_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, obs_channels, 100, 100)
            return self.conv(dummy).view(1, -1).size(1)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return self.fc(self.conv(obs))   # (B, 512)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action_probs, value).

        action_probs: (B, num_actions)  — softmax probabilities
        value:        (B, 1)            — state value estimate
        """
        features = self._encode(obs)
        action_probs = torch.softmax(self.actor_head(features), dim=-1)
        value = self.critic_head(features)
        return action_probs, value

    @torch.no_grad()
    def get_action(
        self, obs: torch.Tensor
    ) -> Tuple[int, float, float]:
        """Sample one action from the policy (agent-side, no gradient).

        Returns:
            action   — int  sampled discrete action
            log_prob — float  log probability of that action
            value    — float  critic estimate V(s)
        """
        action_probs, value = self.forward(obs)
        dist = Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.squeeze(-1).item()),
        )

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log_probs, values and entropy for a batch (trainer-side).

        Args:
            obs:     (B, C, H, W) float32
            actions: (B,)         int64

        Returns:
            log_probs: (B,)  log π(a|s)
            values:    (B, 1) V(s)
            entropy:   (B,)  H[π(·|s)]
        """
        action_probs, values = self.forward(obs)
        dist = Categorical(probs=action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (H, W, C) float32 numpy to (1, C, H, W) float32 tensor."""
    return (
        torch.from_numpy(obs)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
