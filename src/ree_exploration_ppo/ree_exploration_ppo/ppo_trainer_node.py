#!/usr/bin/env python3
"""
PPO Trainer Node — ree_exploration_ppo
========================================
Per-robot PPO trainer (DTDE): one ActorCriticNetwork per robot, fully independent.

  - Receives encoded experience tuples from ppo_agent_node via
    /robot_{id}/ppo_experience
  - Fills a RolloutBuffer (on-policy, fixed-length rollout)
  - When the buffer is full → compute_gae() → PPO-Clip update → clear()
  - Training always in a background thread — NEVER inside ROS2 callbacks
  - Broadcasts updated weights via /robot_{id}/ppo/weight_update after each update
  - Logs scalars to TensorBoard and rows to a CSV file
  - Watchdog: warns if no experience arrives for WATCHDOG_TIMEOUT_SEC seconds
  - Saves atomic checkpoints every 60 s and on SIGTERM

TensorBoard scalars:
  Train/ActorLoss, Train/ValueLoss, Train/EntropyLoss, Train/TotalLoss,
  Train/GradNorm, Train/UpdateStep
  Episode/TotalReward, Episode/TotalReward_MA50, Episode/MineralsDetected,
  Episode/Steps, Eval/AvgReward, Eval/AvgRewardPerStep
"""

import base64
import csv
import json
import os
import signal
import sys
import threading
import time
import zlib
from collections import deque
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import rclpy
import torch
import torch.optim as optim
from rclpy.node import Node
from std_msgs.msg import String

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from .config import NetworkConfig, TopicConfig
from .ppo_networks import ActorCriticNetwork
from .rollout_buffer import RolloutBuffer

_T = TopicConfig()
_N = NetworkConfig()

_CSV_FIELDS = [
    'episode', 'steps', 'total_reward', 'avg_reward',
    'minerals', 'actor_loss', 'value_loss', 'entropy_loss',
    'total_loss', 'grad_norm_avg', 'reward_ma50', 'timestamp',
]


class PPOTrainerNode(Node):
    """Trains one robot's ActorCriticNetwork from on-policy rollouts (background thread)."""

    def __init__(self) -> None:
        super().__init__('ppo_trainer')

        self.declare_parameter('robot_id', 0)
        self._robot_id: int = int(self.get_parameter('robot_id').value)
        self._tag = f'[PPO_TRAINER robot_{self._robot_id}]'

        self._declare_params()
        self._load_params()
        self._init_dirs()

        # Initialize counters to 0 BEFORE loading checkpoint so that
        # _load_checkpoint() can overwrite them with the saved values.
        self._step_count: int = 0
        self._update_count: int = 0
        self._ep_count: int = 0
        self._current_ep: int = -1

        self._init_network()      # calls _load_checkpoint() → overwrites counters
        self._init_buffer()
        self._init_logging()
        self._init_episode_state()

        self._shutdown_event = threading.Event()
        self._train_event    = threading.Event()
        self._last_exp_time  = time.monotonic()

        # ROS2 topics
        exp_topic    = _T.AGENT_EXPERIENCE_FMT.format(id=self._robot_id)
        weight_topic = _T.WEIGHT_UPDATE_FMT.format(id=self._robot_id)

        self.create_subscription(String, exp_topic, self._exp_cb, 100)
        self._weight_pub = self.create_publisher(String, weight_topic, 10)

        self.create_timer(5.0,  self._watchdog_check)
        self.create_timer(60.0, self._periodic_checkpoint)

        self._train_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name=f'ppo-train-{self._robot_id}',
        )
        self._train_thread.start()

        signal.signal(signal.SIGTERM, self._sigterm_handler)

        self.get_logger().info(f'{self._tag} {self._buffer}')
        self.get_logger().info(
            f'{self._tag} Model dir   : {os.path.abspath(self._model_dir)}'
        )
        self.get_logger().info(
            f'{self._tag} TB logs     : tensorboard --logdir '
            f'{os.path.abspath(self._tb_log_dir)}'
        )
        self.get_logger().info(
            f'{self._tag} Exp topic   : {exp_topic}'
        )
        self.get_logger().info(
            f'{self._tag} Weight topic: {weight_topic}'
        )

    # ------------------------------------------------------------------ #
    #  PARAMETERS                                                          #
    # ------------------------------------------------------------------ #

    def _declare_params(self) -> None:
        rid = self._robot_id
        self.declare_parameter('model_dir',        f'{_N.MODEL_BASE_DIR}/robot_{rid}')
        self.declare_parameter('tb_log_dir',       f'{_N.TB_BASE_DIR}/robot_{rid}')
        self.declare_parameter('csv_log_dir',      f'{_N.CSV_BASE_DIR}/robot_{rid}')
        self.declare_parameter('learning_rate',    _N.LEARNING_RATE)
        self.declare_parameter('gamma',            _N.GAMMA)
        self.declare_parameter('clip_eps',         _N.CLIP_EPS)
        self.declare_parameter('n_epochs',         _N.N_EPOCHS)
        self.declare_parameter('mini_batch_size',  _N.MINI_BATCH_SIZE)
        self.declare_parameter('gae_lambda',       _N.GAE_LAMBDA)
        self.declare_parameter('value_coef',       _N.VALUE_COEF)
        self.declare_parameter('entropy_coef',     _N.ENTROPY_COEF)
        self.declare_parameter('rollout_steps',    _N.ROLLOUT_STEPS)
        self.declare_parameter('max_grad_norm',    _N.MAX_GRAD_NORM)
        self.declare_parameter('watchdog_timeout_sec',  _N.WATCHDOG_TIMEOUT_SEC)
        self.declare_parameter('weight_broadcast_freq', _N.WEIGHT_BROADCAST_FREQ)
        self.declare_parameter('target_kl',             _N.TARGET_KL)

    def _load_params(self) -> None:
        self._model_dir    = self.get_parameter('model_dir').value
        self._tb_log_dir   = self.get_parameter('tb_log_dir').value
        self._csv_log_dir  = self.get_parameter('csv_log_dir').value
        self._lr           = float(self.get_parameter('learning_rate').value)
        self._gamma        = float(self.get_parameter('gamma').value)
        self._clip_eps     = float(self.get_parameter('clip_eps').value)
        self._n_epochs     = int(self.get_parameter('n_epochs').value)
        self._mini_batch   = int(self.get_parameter('mini_batch_size').value)
        self._gae_lambda   = float(self.get_parameter('gae_lambda').value)
        self._value_coef   = float(self.get_parameter('value_coef').value)
        self._entropy_coef = float(self.get_parameter('entropy_coef').value)
        self._rollout_steps = int(self.get_parameter('rollout_steps').value)
        self._max_grad_norm = float(self.get_parameter('max_grad_norm').value)
        self._watchdog_timeout = float(
            self.get_parameter('watchdog_timeout_sec').value
        )
        self._weight_bcast_freq = int(
            self.get_parameter('weight_broadcast_freq').value
        )
        self._target_kl = float(self.get_parameter('target_kl').value)

    # ------------------------------------------------------------------ #
    #  SETUP                                                               #
    # ------------------------------------------------------------------ #

    def _init_dirs(self) -> None:
        for d in (self._model_dir, self._tb_log_dir, self._csv_log_dir):
            os.makedirs(d, exist_ok=True)

    def _init_network(self) -> None:
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._net_lock = threading.Lock()

        self._actor_critic = ActorCriticNetwork(
            obs_channels=_N.OBS_CHANNELS, num_actions=_N.NUM_ACTIONS
        ).to(self._device)

        self._optimizer = optim.Adam(
            self._actor_critic.parameters(), lr=self._lr
        )
        self._load_checkpoint()

    def _init_buffer(self) -> None:
        obs_shape     = (_N.MAP_HEIGHT, _N.MAP_WIDTH, _N.OBS_CHANNELS)
        self._buffer  = RolloutBuffer(
            rollout_steps=self._rollout_steps,
            obs_shape=obs_shape,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
        )

    def _init_logging(self) -> None:
        if _TB_AVAILABLE:
            self._tb: Optional[SummaryWriter] = SummaryWriter(
                log_dir=self._tb_log_dir
            )
        else:
            self._tb = None
            self.get_logger().warning(
                f'{self._tag} TensorBoard unavailable (pip install tensorboard)'
            )

        csv_path        = os.path.join(self._csv_log_dir, 'episodes.csv')
        self._csv_path  = csv_path
        write_header    = not os.path.exists(csv_path)
        self._csv_fh    = open(csv_path, 'a', newline='')
        self._csv_writer = csv.DictWriter(
            self._csv_fh, fieldnames=_CSV_FIELDS
        )
        if write_header:
            self._csv_writer.writeheader()

    def _init_episode_state(self) -> None:
        self._ep_reward:   float = 0.0
        self._ep_steps:    int   = 0
        self._ep_minerals: int   = 0
        self._reward_ma: deque = deque(maxlen=50)

        # Protected by _loss_lock; updated by training thread after each update
        self._loss_lock = threading.Lock()
        self._last_update_losses: dict = {
            'actor_loss':   0.0,
            'value_loss':   0.0,
            'entropy_loss': 0.0,
            'grad_norm':    0.0,
        }

    # ------------------------------------------------------------------ #
    #  EXPERIENCE CALLBACK (ROS2 thread — decode + push only)             #
    # ------------------------------------------------------------------ #

    def _exp_cb(self, msg: String) -> None:
        try:
            p      = json.loads(msg.data)
            shape  = tuple(p['obs_shape'])
            obs    = np.frombuffer(
                base64.b64decode(p['obs']), dtype=np.float16
            ).reshape(shape).astype(np.float32)

            log_prob = float(p['log_prob'])
            value    = float(p['value'])

            full = self._buffer.push(
                obs,
                int(p['action']),
                float(p['reward']),
                log_prob,
                value,
                bool(p['done']),
            )

            self._last_exp_time = time.monotonic()
            self._accumulate_episode(p)

            if full:
                # Signal training thread to run PPO update
                self._train_event.set()

        except Exception as exc:
            self.get_logger().warning(
                f'{self._tag} [WARN] _exp_cb: {exc}'
            )

    def _accumulate_episode(self, p: dict) -> None:
        ep     = int(p.get('ep', 0))
        reward = float(p['reward'])

        if ep != self._current_ep:
            if self._current_ep >= 0:
                self._flush_episode()
            self._current_ep  = ep
            self._ep_reward   = 0.0
            self._ep_steps    = 0
            self._ep_minerals = 0

        self._ep_reward += reward
        self._ep_steps  += 1
        # Match agent threshold: cpi > MINERAL_DETECTION_THRESHOLD (0.3)
        # → reward > 0.3 * REWARD_SCALE (5.0) = 1.5
        # (was hardcoded at 15.0 → cpi > 3.0 — inconsistent with agent)
        if reward > _N.MINERAL_DETECTION_THRESHOLD * _N.REWARD_SCALE:
            self._ep_minerals += 1

    def _flush_episode(self) -> None:
        ep      = self._ep_count
        n_steps = max(self._ep_steps, 1)
        avg_rew = self._ep_reward / n_steps

        with self._loss_lock:
            losses = dict(self._last_update_losses)

        actor_avg   = losses['actor_loss']
        value_avg   = losses['value_loss']
        entropy_avg = losses['entropy_loss']
        gnorm_avg   = losses['grad_norm']
        total_avg   = (actor_avg
                       + self._value_coef   * value_avg
                       + self._entropy_coef * entropy_avg)

        self._reward_ma.append(self._ep_reward)
        ma50 = float(np.mean(self._reward_ma))

        if self._tb is not None:
            self._tb.add_scalar('Episode/TotalReward',      self._ep_reward,          ep)
            self._tb.add_scalar('Episode/TotalReward_MA50', ma50,                     ep)
            self._tb.add_scalar('Episode/MineralsDetected', self._ep_minerals,        ep)
            self._tb.add_scalar('Episode/Steps',            self._ep_steps,           ep)
            self._tb.add_scalar('Eval/AvgRewardPerStep',    avg_rew,                  ep)
            self._tb.add_scalar('Eval/MineralRate',
                                self._ep_minerals / max(self._ep_steps, 1),           ep)

        self._csv_writer.writerow({
            'episode':       ep,
            'steps':         self._ep_steps,
            'total_reward':  round(self._ep_reward, 4),
            'avg_reward':    round(avg_rew, 4),
            'minerals':      self._ep_minerals,
            'actor_loss':    round(actor_avg, 6),
            'value_loss':    round(value_avg, 6),
            'entropy_loss':  round(entropy_avg, 6),
            'total_loss':    round(total_avg, 6),
            'grad_norm_avg': round(gnorm_avg, 4),
            'reward_ma50':   round(ma50, 4),
            'timestamp':     time.strftime('%Y-%m-%dT%H:%M:%S'),
        })
        self._csv_fh.flush()

        self._ep_count    += 1
        self._ep_reward    = 0.0
        self._ep_steps     = 0
        self._ep_minerals  = 0

    # ------------------------------------------------------------------ #
    #  TRAINING LOOP (background thread)                                   #
    # ------------------------------------------------------------------ #

    def _training_loop(self) -> None:
        self.get_logger().info(f'{self._tag} Training thread started')
        while not self._shutdown_event.is_set():
            # Block until the rollout buffer signals it is full
            triggered = self._train_event.wait(timeout=5.0)
            if not triggered or self._shutdown_event.is_set():
                continue
            self._train_event.clear()

            # Bootstrap V(s_{T+1}): if the rollout ended mid-episode (done=False),
            # use the critic estimate at the last step rather than 0.  Using 0
            # when the episode is ongoing underestimates future returns and
            # introduces a systematic negative bias in GAE advantages.
            last_val = 0.0 if self._buffer.last_done else self._buffer.last_value
            self._buffer.compute_gae(last_value=last_val)

            actor_losses, value_losses, entropy_losses = [], [], []
            grad_norms, kls, clip_fracs = [], [], []

            for epoch in range(self._n_epochs):
                epoch_kls: list = []
                for batch in self._buffer.get_batches(self._mini_batch):
                    al, vl, el, gn, kl, cf = self._train_step(batch)
                    actor_losses.append(al)
                    value_losses.append(vl)
                    entropy_losses.append(el)
                    grad_norms.append(gn)
                    kls.append(kl)
                    clip_fracs.append(cf)
                    epoch_kls.append(kl)

                # KL early stopping (Schulman blog): if the mean KL across this
                # epoch already exceeds the target, further epochs would push the
                # policy too far from the old one → stop to avoid instability.
                if float(np.mean(epoch_kls)) > self._target_kl:
                    self.get_logger().info(
                        f'{self._tag} KL early stop at epoch '
                        f'{epoch + 1}/{self._n_epochs} '
                        f'(KL={float(np.mean(epoch_kls)):.4f} > {self._target_kl})'
                    )
                    break

            self._buffer.clear()

            al_mean  = float(np.mean(actor_losses))
            vl_mean  = float(np.mean(value_losses))
            el_mean  = float(np.mean(entropy_losses))
            gn_mean  = float(np.mean(grad_norms))
            kl_mean  = float(np.mean(kls))
            cf_mean  = float(np.mean(clip_fracs))

            self._update_count += 1
            self._step_count   += len(actor_losses)

            with self._loss_lock:
                self._last_update_losses = {
                    'actor_loss':   al_mean,
                    'value_loss':   vl_mean,
                    'entropy_loss': el_mean,
                    'grad_norm':    gn_mean,
                }

            self._log_update(al_mean, vl_mean, el_mean, gn_mean, kl_mean, cf_mean)

            self._broadcast_weights()

    def _train_step(
        self, batch: Tuple[np.ndarray, ...]
    ) -> Tuple[float, float, float, float, float, float]:
        """One PPO-Clip mini-batch gradient step.

        Returns (actor_loss, value_loss, entropy_loss, grad_norm, kl, clip_frac).

        Value loss is clipped (SB3/CleanRL style) to prevent the critic from
        making excessively large updates that would flood the shared CNN gradients
        and destroy the actor's feature representation.
        """
        obs_np, actions_np, old_log_probs_np, advantages_np, returns_np, old_values_np = batch

        obs_t        = torch.FloatTensor(obs_np).permute(0, 3, 1, 2).to(self._device)
        actions_t    = torch.LongTensor(actions_np).to(self._device)
        old_lp_t     = torch.FloatTensor(old_log_probs_np).to(self._device)
        adv_t        = torch.FloatTensor(advantages_np).to(self._device)
        returns_t    = torch.FloatTensor(returns_np).to(self._device)
        old_values_t = torch.FloatTensor(old_values_np).to(self._device)

        # ── Normalize returns per mini-batch ──────────────────────────────
        # Raw returns have large magnitude (e.g. -83 for all-penalty episodes).
        # Normalizing brings ValueLoss ≈ 1.0 so VALUE_COEF=0.5 gives a balanced
        # gradient split between actor and critic (was: value=99.9% of total loss).
        ret_mean = returns_t.mean()
        ret_std  = (returns_t.std() + 1e-8)
        returns_t_norm    = (returns_t    - ret_mean) / ret_std
        old_values_t_norm = (old_values_t - ret_mean) / ret_std

        with self._net_lock:
            log_probs_new, values_new, entropy = self._actor_critic.evaluate(
                obs_t, actions_t
            )
            values_new_sq = values_new.squeeze(-1)

            # ── Actor loss (PPO-Clip) ──────────────────────────────────────
            ratio  = torch.exp(log_probs_new - old_lp_t)
            surr1  = ratio * adv_t
            surr2  = torch.clamp(ratio, 1.0 - self._clip_eps, 1.0 + self._clip_eps) * adv_t
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            # ── Clipped value loss on normalized returns (SB3 / CleanRL) ──
            # Using normalized targets keeps ValueLoss ≈ 1.0 regardless of
            # reward scale, preventing the critic from flooding shared-CNN grads.
            v_clipped = old_values_t_norm + torch.clamp(
                values_new_sq - old_values_t_norm,
                -self._clip_eps, self._clip_eps,
            )
            vf_loss1 = torch.nn.functional.mse_loss(values_new_sq, returns_t_norm)
            vf_loss2 = torch.nn.functional.mse_loss(v_clipped,     returns_t_norm)
            value_loss = torch.max(vf_loss1, vf_loss2)

            # ── Entropy loss ──────────────────────────────────────────────
            entropy_loss = -torch.mean(entropy)

            loss = actor_loss + self._value_coef * value_loss + self._entropy_coef * entropy_loss

            self._optimizer.zero_grad()
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._actor_critic.parameters(), self._max_grad_norm
                ).item()
            )
            self._optimizer.step()

        # ── Diagnostics (no grad needed) ──────────────────────────────────
        with torch.no_grad():
            # Approx KL: mean of (old_log_prob - new_log_prob)  (Schulman blog)
            kl = float(torch.mean(old_lp_t - log_probs_new).item())
            # Fraction of samples where the ratio was clipped
            clip_frac = float(
                torch.mean((torch.abs(ratio - 1.0) > self._clip_eps).float()).item()
            )

        return (
            float(actor_loss.item()),
            float(value_loss.item()),
            float(entropy_loss.item()),
            grad_norm,
            kl,
            clip_frac,
        )

    def _log_update(
        self,
        actor_loss: float,
        value_loss: float,
        entropy_loss: float,
        grad_norm: float,
        kl: float,
        clip_frac: float,
    ) -> None:
        if self._tb is None:
            return
        u = self._update_count
        total = actor_loss + self._value_coef * value_loss + self._entropy_coef * entropy_loss
        self._tb.add_scalar('Train/ActorLoss',    actor_loss,   u)
        self._tb.add_scalar('Train/ValueLoss',    value_loss,   u)
        self._tb.add_scalar('Train/EntropyLoss',  entropy_loss, u)
        self._tb.add_scalar('Train/TotalLoss',    total,        u)
        self._tb.add_scalar('Train/GradNorm',     grad_norm,    u)
        self._tb.add_scalar('Train/ApproxKL',     kl,           u)
        self._tb.add_scalar('Train/ClipFraction', clip_frac,    u)
        self._tb.add_scalar('Train/UpdateStep',   u,            u)

    # ------------------------------------------------------------------ #
    #  WEIGHT BROADCAST                                                    #
    # ------------------------------------------------------------------ #

    def _broadcast_weights(self) -> None:
        """Serialize actor_critic → zlib compress → base64 → publish String."""
        try:
            buf = BytesIO()
            with self._net_lock:
                torch.save(self._actor_critic.state_dict(), buf)
            compressed = zlib.compress(buf.getvalue(), level=1)
            encoded    = base64.b64encode(compressed).decode('ascii')
            msg        = String()
            msg.data   = encoded
            self._weight_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(
                f'{self._tag} [WARN] _broadcast_weights: {exc}'
            )

    # ------------------------------------------------------------------ #
    #  CHECKPOINT                                                          #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self) -> None:
        """Atomically save actor_critic, optimizer, and counters."""
        path = os.path.join(self._model_dir, 'latest.pt')
        tmp  = path + '.tmp'
        with self._net_lock:
            payload = {
                'actor_critic_state_dict': self._actor_critic.state_dict(),
                'optimizer_state_dict':    self._optimizer.state_dict(),
                'update_count':            self._update_count,
                'step_count':              self._step_count,
                'episode_count':           self._ep_count,
            }
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self.get_logger().info(
            f'{self._tag} Checkpoint saved — '
            f'ep={self._ep_count} updates={self._update_count}'
        )

    def _load_checkpoint(self) -> None:
        path = os.path.join(self._model_dir, 'latest.pt')
        if not os.path.exists(path):
            self.get_logger().info(f'{self._tag} No checkpoint — starting fresh')
            return
        try:
            ckpt = torch.load(
                path, map_location=self._device, weights_only=True
            )
            self._actor_critic.load_state_dict(ckpt['actor_critic_state_dict'])
            self._optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self._update_count = int(ckpt.get('update_count', 0))
            self._step_count   = int(ckpt.get('step_count', 0))
            self._ep_count     = int(ckpt.get('episode_count', 0))
            self.get_logger().info(
                f'{self._tag} Checkpoint loaded — '
                f'ep={self._ep_count} updates={self._update_count}'
            )
        except Exception as exc:
            self.get_logger().error(
                f'{self._tag} [ERROR] _load_checkpoint: {exc}'
            )

    def _periodic_checkpoint(self) -> None:
        if self._update_count > 0:
            self._save_checkpoint()

    # ------------------------------------------------------------------ #
    #  WATCHDOG                                                            #
    # ------------------------------------------------------------------ #

    def _watchdog_check(self) -> None:
        elapsed = time.monotonic() - self._last_exp_time
        if elapsed > self._watchdog_timeout:
            self.get_logger().warning(
                f'{self._tag} [WARN] No experience for {elapsed:.0f}s — '
                'agent may be stalled'
            )

    # ------------------------------------------------------------------ #
    #  LIFECYCLE                                                           #
    # ------------------------------------------------------------------ #

    def _sigterm_handler(self, signum, frame) -> None:
        self.get_logger().info(
            f'{self._tag} SIGTERM — saving checkpoint'
        )
        self._shutdown_event.set()
        self._train_event.set()   # unblock training thread
        if self._update_count > 0:
            self._save_checkpoint()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    def destroy_node(self) -> None:
        self._shutdown_event.set()
        self._train_event.set()
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass
        try:
            self._csv_fh.close()
        except Exception:
            pass
        super().destroy_node()


def main(argv: Optional[list] = None) -> None:
    rclpy.init(args=argv)
    node = PPOTrainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
