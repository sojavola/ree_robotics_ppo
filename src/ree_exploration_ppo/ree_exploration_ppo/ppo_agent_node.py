#!/usr/bin/env python3
"""
PPO Agent Node — ree_exploration_ppo
======================================
Lightweight ROS2 node implementing the agent-side of DTDE PPO:

  observe  — build global 100×100×C observation (all channels)
  decide   — sample action from π_θ (ActorCriticNetwork), record log_prob + value
  act      — move in grid, compute reward (article Eq. 1-2, Caccavale 2023)
  publish  — position, cleaning action, experience (JSON+b64), telemetry

Training runs in ppo_trainer_node (separate process, one per robot).
Updated weights arrive via /robot_{id}/ppo/weight_update.
SIGTERM triggers a clean shutdown.
"""

import base64
import json
import signal
import sys
import threading
import zlib
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32, String

import torch

from .config import NetworkConfig, TopicConfig
from .ppo_networks import ActorCriticNetwork, obs_to_tensor
from .reward_system import RealMineralRewardSystem

_T = TopicConfig()
_N = NetworkConfig()

# 8 chess-king moves: N S W E NW NE SW SE (article: |A|=8)
_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, -1), (-1, 0), (1, 0),
    (-1, 1), (1, 1), (-1, -1), (1, -1),
)


class PPOAgentNode(Node):
    """DTDE agent: observe (global 100×100) → decide → act → publish."""

    def __init__(self, robot_id: int = 0) -> None:
        super().__init__('ppo_agent')
        self.declare_parameter('robot_id', robot_id)
        self.robot_id = int(self.get_parameter('robot_id').value)

        self._exp_topic    = _T.AGENT_EXPERIENCE_FMT.format(id=self.robot_id)
        self._weight_topic = _T.WEIGHT_UPDATE_FMT.format(id=self.robot_id)

        self._init_state()
        self._init_network()
        self._init_publishers()
        self._init_subscribers()
        self._init_timers()
        self._register_sigterm()

        self.get_logger().info(
            f'[PPO_AGENT_{self.robot_id}] Ready — '
            f'pos={self._pos} device={self._device}'
        )

    # ------------------------------------------------------------------ #
    #  INITIALISATION                                                      #
    # ------------------------------------------------------------------ #

    def _init_state(self) -> None:
        self._map_lock = threading.Lock()

        self._mineral_map = np.zeros(
            (_N.MAP_HEIGHT, _N.MAP_WIDTH, _N.MINERAL_CHANNELS),
            dtype=np.float32,
        )
        self._obstacle_map = np.zeros(
            (_N.MAP_HEIGHT, _N.MAP_WIDTH), dtype=np.float32
        )
        self._visited = np.zeros(
            (_N.MAP_HEIGHT, _N.MAP_WIDTH), dtype=np.float32
        )

        m = _N.SPAWN_MARGIN
        self._pos: Tuple[int, int] = (
            int(np.random.randint(m, _N.MAP_WIDTH - m)),
            int(np.random.randint(m, _N.MAP_HEIGHT - m)),
        )

        # Reward system — identique à QMIX pour benchmark équitable
        self._reward_system = RealMineralRewardSystem(
            grid_size=(_N.MAP_HEIGHT, _N.MAP_WIDTH),
            robot_id=self.robot_id,
            coverage_weight=0.5,
        )

        # PPO per-step state: set by _decide(), consumed by _publish_experience()
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0

        self._step: int = 0
        self._episode: int = 0
        self._ep_reward: float = 0.0
        self._ep_minerals: int = 0

    def _init_network(self) -> None:
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._net_lock = threading.Lock()
        self._actor_critic = ActorCriticNetwork(
            obs_channels=_N.OBS_CHANNELS, num_actions=_N.NUM_ACTIONS
        ).to(self._device)
        self._actor_critic.eval()

    def _init_publishers(self) -> None:
        self._pos_pub = self.create_publisher(
            Pose2D, f'/robot_{self.robot_id}/position', 10
        )
        self._vel_pub = self.create_publisher(
            Twist, f'/robot_{self.robot_id}/cmd_vel', 10
        )
        self._clean_pub = self.create_publisher(
            Float32MultiArray,
            f'/robot_{self.robot_id}/cleaning_action', 10,
        )
        self._exp_pub   = self.create_publisher(String, self._exp_topic, 10)
        self._step_pub  = self.create_publisher(Int32, _T.AGENT_STEP, 10)
        self._disc_pub  = self.create_publisher(
            Float32MultiArray, _T.SHARED_DISCOVERIES, 10
        )
        self._status_pub = self.create_publisher(
            String, f'/robot_{self.robot_id}/status', 10
        )
        self._reset_pub = self.create_publisher(String, _T.EPISODE_RESET, 10)

    def _init_subscribers(self) -> None:
        self.create_subscription(
            Float32MultiArray, _T.MINERAL_MAP, self._mineral_cb, 10
        )
        self.create_subscription(
            OccupancyGrid, _T.OBSTACLE_MAP, self._obstacle_cb, 10
        )
        self.create_subscription(
            String, self._weight_topic, self._weight_cb, 10
        )

    def _init_timers(self) -> None:
        self.create_timer(1.0 / _N.DECISION_HZ, self._decision_step)
        self.create_timer(0.5, self._publish_position)
        self.create_timer(5.0, self._publish_status)

    def _register_sigterm(self) -> None:
        signal.signal(signal.SIGTERM, self._sigterm_handler)

    # ------------------------------------------------------------------ #
    #  MAIN STEP                                                           #
    # ------------------------------------------------------------------ #

    def _decision_step(self) -> None:
        """observe → decide → act → publish."""
        try:
            obs    = self._observe()
            action = self._decide(obs)
            reward, _ = self._act(action)

            # Episode ends exclusively at step 300 — done=True only on the
            # final step so GAE correctly cuts the bootstrap at that boundary.
            done = (self._step + 1) >= _N.MAX_STEPS_PER_EPISODE

            self._publish_experience(obs, action, reward, done)
            self._publish_telemetry()

            self._ep_reward += reward
            self._step += 1

            if done:
                self._handle_episode_end()

        except Exception as exc:
            self.get_logger().error(
                f'[PPO_AGENT_{self.robot_id}] [ERROR] _decision_step: {exc}'
            )

    # ------------------------------------------------------------------ #
    #  OBSERVE — global 100×100×OBS_CHANNELS map                         #
    # ------------------------------------------------------------------ #

    def _observe(self) -> np.ndarray:
        """Build 6-channel observation (4 REE + obstacles + exploration).

        Returns: (MAP_HEIGHT, MAP_WIDTH, 6) float32
        """
        with self._map_lock:
            mineral  = self._mineral_map.copy()
            obstacle = self._obstacle_map.copy()
            visited  = self._visited.copy()

        m_max = np.max(mineral)
        if m_max > 1e-8:
            mineral = mineral / m_max

        obs_obstacle = np.clip(obstacle / 100.0, 0.0, 1.0).reshape(
            _N.MAP_HEIGHT, _N.MAP_WIDTH, 1
        )
        obs_visited = visited.reshape(_N.MAP_HEIGHT, _N.MAP_WIDTH, 1)

        return np.concatenate(
            [mineral, obs_obstacle, obs_visited], axis=-1
        ).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  DECIDE — stochastic policy sampling                                 #
    # ------------------------------------------------------------------ #

    def _decide(self, obs: np.ndarray) -> int:
        """Sample action from π_θ(a|s); store log_prob and value for experience."""
        try:
            obs_t = obs_to_tensor(obs, self._device)
            with self._net_lock:
                action, log_prob, value = self._actor_critic.get_action(obs_t)
            self._last_log_prob = log_prob
            self._last_value    = value
            return action
        except Exception as exc:
            self.get_logger().warning(
                f'[PPO_AGENT_{self.robot_id}] [WARN] _decide: {exc}'
            )
            return int(np.random.randint(0, _N.NUM_ACTIONS))

    # ------------------------------------------------------------------ #
    #  ACT — reward from article Eq. 1-2                                  #
    # ------------------------------------------------------------------ #

    def _act(self, action: int) -> Tuple[float, bool]:
        """Move one cell, compute reward via RealMineralRewardSystem (QMIX-aligned)."""
        x, y   = self._pos
        dx, dy = _DIRECTIONS[int(action) % len(_DIRECTIONS)]
        nx     = int(np.clip(x + dx, 0, _N.MAP_WIDTH - 1))
        ny     = int(np.clip(y + dy, 0, _N.MAP_HEIGHT - 1))

        if not self._is_valid(nx, ny):
            # Collision penalty — identique à QMIX collision_penalty
            return self._reward_system.reward_config['collision_penalty'], False

        self._pos = (nx, ny)
        with self._map_lock:
            self._visited[ny, nx] = min(self._visited[ny, nx] + 0.5, 1.0)
            mineral = self._mineral_map[ny, nx, :].tolist()

        is_new = (nx, ny) not in self._reward_system.visited_positions
        reward = self._reward_system.calculate_reward(
            mineral_concentrations=mineral,
            position=(nx, ny),
            is_new_position=is_new,
            has_collision=False,
            step_count=self._step,
        )

        self._ep_minerals = self._reward_system.minerals_collected

        self._publish_cleaning_action(nx, ny)
        self._publish_velocity()
        if reward > 15.0:
            self._publish_discovery(nx, ny, reward)

        return reward, False

    # ------------------------------------------------------------------ #
    #  EPISODE LIFECYCLE                                                   #
    # ------------------------------------------------------------------ #

    def _handle_episode_end(self) -> None:
        self._episode += 1
        self.get_logger().info(
            f'[PPO_AGENT_{self.robot_id}] [INFO] '
            f'ep={self._episode} steps={self._step} '
            f'reward={self._ep_reward:.2f} minerals={self._ep_minerals}'
        )
        if self.robot_id == 0:
            msg      = String()
            msg.data = str(self._episode)
            self._reset_pub.publish(msg)
        self._reset_episode()

    def _reset_episode(self) -> None:
        m = _N.SPAWN_MARGIN
        self._pos = (
            int(np.random.randint(m, _N.MAP_WIDTH - m)),
            int(np.random.randint(m, _N.MAP_HEIGHT - m)),
        )
        with self._map_lock:
            self._visited[:] = 0.0
        self._reward_system.reset_episode()
        self._step        = 0
        self._ep_reward   = 0.0
        self._ep_minerals = 0

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _is_valid(self, x: int, y: int) -> bool:
        with self._map_lock:
            if not (0 <= x < _N.MAP_WIDTH and 0 <= y < _N.MAP_HEIGHT):
                return False
            return int(self._obstacle_map[y, x]) <= _N.OBSTACLE_THRESHOLD

    # ------------------------------------------------------------------ #
    #  ROS2 CALLBACKS                                                      #
    # ------------------------------------------------------------------ #

    def _mineral_cb(self, msg: Float32MultiArray) -> None:
        data     = np.array(msg.data, dtype=np.float32)
        expected = _N.MAP_HEIGHT * _N.MAP_WIDTH * _N.MINERAL_CHANNELS
        if data.size != expected:
            return
        with self._map_lock:
            self._mineral_map = data.reshape(
                _N.MAP_HEIGHT, _N.MAP_WIDTH, _N.MINERAL_CHANNELS
            )

    def _obstacle_cb(self, msg: OccupancyGrid) -> None:
        data = np.array(msg.data, dtype=np.float32)
        if data.size == _N.MAP_HEIGHT * _N.MAP_WIDTH:
            with self._map_lock:
                self._obstacle_map = data.reshape(_N.MAP_HEIGHT, _N.MAP_WIDTH)

    def _weight_cb(self, msg: String) -> None:
        """Decompress actor_critic weights from trainer and apply locally."""
        try:
            compressed = base64.b64decode(msg.data.encode('ascii'))
            raw        = zlib.decompress(compressed)
            state_dict = torch.load(
                BytesIO(raw), map_location=self._device, weights_only=True
            )
            with self._net_lock:
                self._actor_critic.load_state_dict(state_dict)
            self._actor_critic.eval()
        except Exception as exc:
            self.get_logger().warning(
                f'[PPO_AGENT_{self.robot_id}] [WARN] _weight_cb: {exc}'
            )

    # ------------------------------------------------------------------ #
    #  PUBLISHERS                                                          #
    # ------------------------------------------------------------------ #

    def _publish_experience(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """JSON + base64 float16 — PPO payload includes log_prob + value."""
        payload = {
            'robot_id':  self.robot_id,
            'obs':       base64.b64encode(
                obs.astype(np.float16).tobytes()
            ).decode('ascii'),
            'obs_shape': list(obs.shape),
            'action':    int(action),
            'reward':    float(reward),
            'log_prob':  float(self._last_log_prob),
            'value':     float(self._last_value),
            'done':      bool(done),
            'ep':        self._episode,
            'step':      self._step,
        }
        msg      = String()
        msg.data = json.dumps(payload, separators=(',', ':'))
        self._exp_pub.publish(msg)

    def _publish_telemetry(self) -> None:
        step_msg      = Int32()
        step_msg.data = int(self._step)
        self._step_pub.publish(step_msg)

    def _publish_position(self) -> None:
        msg       = Pose2D()
        msg.x     = float(self._pos[0])
        msg.y     = float(self._pos[1])
        msg.theta = 0.0
        self._pos_pub.publish(msg)

    def _publish_cleaning_action(self, x: int, y: int) -> None:
        msg      = Float32MultiArray()
        msg.data = [float(x), float(y)]
        self._clean_pub.publish(msg)

    def _publish_velocity(self) -> None:
        msg              = Twist()
        msg.linear.x     = float(_N.CMD_VEL_LINEAR)
        self._vel_pub.publish(msg)

    def _publish_discovery(self, x: int, y: int, reward: float) -> None:
        with self._map_lock:
            mineral_data = self._mineral_map[y, x, :].tolist()
        msg      = Float32MultiArray()
        msg.data = (
            [float(self.robot_id), float(x), float(y), float(reward)]
            + mineral_data
        )
        self._disc_pub.publish(msg)

    def _publish_status(self) -> None:
        text     = (
            f'PPO Agent {self.robot_id} | '
            f'ep={self._episode} step={self._step} '
            f'reward={self._ep_reward:.1f}'
        )
        msg      = String()
        msg.data = text
        self._status_pub.publish(msg)

    # ------------------------------------------------------------------ #
    #  LIFECYCLE                                                           #
    # ------------------------------------------------------------------ #

    def _sigterm_handler(self, signum, frame) -> None:
        self.get_logger().info(
            f'[PPO_AGENT_{self.robot_id}] SIGTERM — shutting down'
        )
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)


def main(argv: Optional[list] = None) -> None:
    rclpy.init(args=argv)
    node = PPOAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
