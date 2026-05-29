"""
On-policy Rollout Buffer with GAE-Lambda for PPO — ree_exploration_ppo.

Collects a fixed-length rollout of ROLLOUT_STEPS transitions, then computes
Generalized Advantage Estimation (GAE, Schulman et al. 2016) in a single
vectorized pass before yielding mini-batches for PPO updates.

Storage arrays (pre-allocated, size = ROLLOUT_STEPS):
    _obs:       (T, H, W, C)  float16
    _actions:   (T,)          int8
    _rewards:   (T,)          float32
    _log_probs: (T,)          float32
    _values:    (T,)          float32
    _dones:     (T,)          bool

After compute_gae():
    _advantages: (T,)  float32  (normalized)
    _returns:    (T,)  float32

Thread-safe: push() and clear() are protected by a Lock.
get_batches() is called only from the training thread (after clear is done).
"""

import threading
from typing import Iterator, List, Tuple

import numpy as np


class RolloutBuffer:
    """Fixed-size on-policy rollout buffer with GAE and mini-batch iteration."""

    def __init__(
        self,
        rollout_steps: int,
        obs_shape: Tuple[int, ...],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self._T = rollout_steps
        self._obs_shape = obs_shape
        self._gamma = gamma
        self._lam = gae_lambda
        self._lock = threading.Lock()

        # Pre-allocated storage (float16 for obs to halve memory)
        self._obs       = np.zeros((rollout_steps, *obs_shape), dtype=np.float16)
        self._actions   = np.zeros(rollout_steps, dtype=np.int8)
        self._rewards   = np.zeros(rollout_steps, dtype=np.float32)
        self._log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self._values    = np.zeros(rollout_steps, dtype=np.float32)
        self._dones     = np.zeros(rollout_steps, dtype=np.bool_)

        # Filled by compute_gae()
        self._advantages = np.zeros(rollout_steps, dtype=np.float32)
        self._returns    = np.zeros(rollout_steps, dtype=np.float32)

        self._ptr: int = 0
        self._full: bool = False

    # ------------------------------------------------------------------ #
    #  WRITE                                                               #
    # ------------------------------------------------------------------ #

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> bool:
        """Store one transition. Returns True when the buffer just became full.

        Drops the transition silently if the buffer is already full (i.e. the
        training thread has not called clear() yet).  This is the correct
        on-policy behaviour: we never overwrite the rollout we are about to
        train on, and we never write beyond the pre-allocated arrays.
        """
        with self._lock:
            if self._full:          # training thread hasn't called clear() yet
                return False
            i = self._ptr
            self._obs[i]       = obs.astype(np.float16)
            self._actions[i]   = int(action)
            self._rewards[i]   = float(reward)
            self._log_probs[i] = float(log_prob)
            self._values[i]    = float(value)
            self._dones[i]     = bool(done)
            self._ptr += 1
            if self._ptr >= self._T:
                self._full = True
                return True
            return False

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._full

    @property
    def last_done(self) -> bool:
        """True if the final stored transition ended the episode (done=True)."""
        with self._lock:
            return bool(self._dones[self._ptr - 1]) if self._ptr > 0 else True

    @property
    def last_value(self) -> float:
        """V(s_T): critic estimate at the last stored step — used as GAE bootstrap."""
        with self._lock:
            return float(self._values[self._ptr - 1]) if self._ptr > 0 else 0.0

    def __len__(self) -> int:
        with self._lock:
            return self._ptr

    # ------------------------------------------------------------------ #
    #  GAE COMPUTATION                                                     #
    # ------------------------------------------------------------------ #

    def compute_gae(self, last_value: float = 0.0) -> None:
        """Compute advantages via GAE-Lambda and returns = A + V(s).

        Must be called from the training thread after the buffer is full.
        last_value: V(s_{T+1}) — 0 if episode ended, else bootstrap estimate.

        GAE formula:
            δ_t  = r_t + γ·V(s_{t+1})·(1-d_t) - V(s_t)
            Â_t  = Σ_{k≥0} (γλ)^k · δ_{t+k}
        Normalised: Â ← (Â - mean(Â)) / (std(Â) + 1e-8)
        Returns:  G_t = Â_t + V(s_t)
        """
        T = self._ptr   # may be < self._T if called before full (shouldn't happen)
        gae = 0.0
        next_value = last_value

        advantages = self._advantages
        returns    = self._returns

        for t in reversed(range(T)):
            mask        = 1.0 - float(self._dones[t])
            delta       = (
                self._rewards[t]
                + self._gamma * next_value * mask
                - self._values[t]
            )
            gae         = delta + self._gamma * self._lam * mask * gae
            advantages[t] = gae
            next_value  = self._values[t]

        returns[:T] = advantages[:T] + self._values[:T]

        # Normalize advantages over the full rollout
        adv = advantages[:T]
        advantages[:T] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ------------------------------------------------------------------ #
    #  MINI-BATCH ITERATION                                                #
    # ------------------------------------------------------------------ #

    def get_batches(
        self, mini_batch_size: int
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        """Yield shuffled mini-batches as numpy arrays.

        Each batch is a tuple:
            (obs_f32, actions_i64, old_log_probs, advantages, returns, old_values)

        old_values is required by the clipped value loss in the trainer.
        Called only from training thread; no lock needed (buffer is static here).
        """
        T = self._ptr
        indices = np.random.permutation(T)

        obs_f32 = self._obs[:T].astype(np.float32)

        for start in range(0, T, mini_batch_size):
            idx = indices[start: start + mini_batch_size]
            yield (
                obs_f32[idx],
                self._actions[idx].astype(np.int64),
                self._log_probs[idx].copy(),
                self._advantages[idx].copy(),
                self._returns[idx].copy(),
                self._values[idx].copy(),     # old V(s) for clipped value loss
            )

    # ------------------------------------------------------------------ #
    #  RESET                                                               #
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Reset buffer pointer. Called after training update is complete."""
        with self._lock:
            self._ptr  = 0
            self._full = False

    # ------------------------------------------------------------------ #
    #  DUNDER                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"RolloutBuffer(ptr={self._ptr}/{self._T}, "
            f"obs={self._obs_shape}, γ={self._gamma}, λ={self._lam})"
        )
