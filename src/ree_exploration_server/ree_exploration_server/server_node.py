#!/usr/bin/env python3

import threading
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.ndimage import gaussian_filter
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid

from .configs import MapConfig, ServerTopicConfig

try:
    from .advanced_mineral_generator import AdvancedMineralGenerator
    _ADVANCED_AVAILABLE = True
except ImportError as exc:
    _ADVANCED_AVAILABLE = False
    print(f"[SERVER] [WARN] AdvancedMineralGenerator unavailable: {exc}")

_TOPICS = ServerTopicConfig()
_MAP = MapConfig()


# ============================================================
#  BASIC (FALLBACK) MINERAL GENERATOR
# ============================================================

class MineralMapGenerator:
    """Procedural mineral map generator — used when advanced generator is absent."""

    def __init__(self, width: int = 100, height: int = 100) -> None:
        self.width = width
        self.height = height

    def generate_mineral_map(self) -> np.ndarray:
        mineral_map = np.zeros((self.height, self.width, 4))
        for ch in range(4):
            base = np.random.rand(self.height, self.width)
            filtered = gaussian_filter(base, sigma=3.0)
            threshold = 0.6 + np.random.random() * 0.3
            deposits = np.where(filtered > threshold, filtered, 0.0)
            if np.max(deposits) > 0:
                deposits = deposits / np.max(deposits)
            mineral_map[:, :, ch] = deposits
        return mineral_map


# ============================================================
#  SERVER NODE
# ============================================================

class REEExplorationServer(Node):
    def __init__(self) -> None:
        super().__init__('ree_exploration_server')

        self.declare_parameter('use_advanced_generator', True)
        use_advanced = (
            self.get_parameter('use_advanced_generator')
            .get_parameter_value()
            .bool_value
        )

        if use_advanced and _ADVANCED_AVAILABLE:
            self.mineral_generator = AdvancedMineralGenerator(
                _MAP.WIDTH, _MAP.HEIGHT
            )
            self.generator_type = 'advanced'
        else:
            self.mineral_generator = MineralMapGenerator(_MAP.WIDTH, _MAP.HEIGHT)
            self.generator_type = 'basic'

        self.get_logger().info(
            f'[SERVER] [INFO] Using {self.generator_type.upper()} mineral generator'
        )

        # State
        self.mineral_map: Optional[np.ndarray] = None
        self.underground_layers: list = []
        self.robot_positions: dict = {}
        self.exploration_map = np.zeros((_MAP.HEIGHT, _MAP.WIDTH))
        self.obstacle_map = np.zeros((_MAP.HEIGHT, _MAP.WIDTH))
        self.episode_count: int = 0

        # Counters for periodic logging
        self.map_publish_count: int = 0
        self.status_counter: int = 0
        self.underground_publish_count: int = 0

        self.lock = threading.Lock()

        # Publishers
        self.mineral_pub = self.create_publisher(
            Float32MultiArray, _TOPICS.MINERAL_MAP, 10
        )
        self.obstacle_pub = self.create_publisher(
            OccupancyGrid, _TOPICS.OBSTACLE_MAP, 10
        )
        self.science_pub = self.create_publisher(
            Float32MultiArray, _TOPICS.SCIENCE_TARGETS, 10
        )
        self.status_pub = self.create_publisher(String, _TOPICS.SYSTEM_STATUS, 10)

        if self.generator_type == 'advanced':
            self.underground_pub = self.create_publisher(
                Float32MultiArray, _TOPICS.UNDERGROUND_LAYERS, 10
            )

        # Subscriptions
        self.episode_reset_sub = self.create_subscription(
            String, _TOPICS.EPISODE_RESET, self._episode_reset_callback, 10
        )

        for i in range(_MAP.NUM_ROBOTS):
            self.create_subscription(
                Pose2D,
                f'/robot_{i}/position',
                self._make_position_callback(i),
                10,
            )
            self.create_subscription(
                Float32MultiArray,
                f'/robot_{i}/cleaning_action',
                self._make_cleaning_callback(i),
                10,
            )

        # Timers
        self.map_timer = self.create_timer(
            _MAP.MAP_PUBLISH_PERIOD, self._publish_maps
        )
        self.status_timer = self.create_timer(
            _MAP.STATUS_PUBLISH_PERIOD, self._publish_status
        )
        if self.generator_type == 'advanced':
            self.underground_timer = self.create_timer(
                _MAP.UNDERGROUND_PUBLISH_PERIOD, self._publish_underground_layers
            )

        self._initialize_system()

        self.get_logger().info(
            f'[SERVER] [INFO] Node initialized — '
            f'map={_MAP.WIDTH}x{_MAP.HEIGHT} robots={_MAP.NUM_ROBOTS} '
            f'generator={self.generator_type.upper()}'
        )

    # ------------------------------------------------------------------ #
    #  INITIALISATION                                                      #
    # ------------------------------------------------------------------ #

    def _initialize_system(self) -> None:
        with self.lock:
            if self.generator_type == 'advanced':
                self.mineral_map = self.mineral_generator.generate_geological_map()
                self.underground_layers = (
                    self.mineral_generator.generate_underground_layers(
                        self.mineral_map, num_layers=3
                    )
                )
                self._log_mineral_clusters()
            else:
                self.mineral_map = self.mineral_generator.generate_mineral_map()

            self._generate_obstacles()

            for i in range(_MAP.NUM_ROBOTS):
                self.robot_positions[i] = self._get_valid_start_position()

        self.get_logger().info('[SERVER] [INFO] System initialized')

    def _log_mineral_clusters(self) -> None:
        """Log mineral cluster statistics (advanced generator only)."""
        if self.generator_type != 'advanced':
            return
        mineral_names = [
            'REE_Oxides', 'REE_Silicates', 'REE_Phosphates', 'REE_Carbonates'
        ]
        total = 0
        for idx, name in enumerate(mineral_names):
            clusters = self.mineral_generator.detect_mineral_clusters(
                self.mineral_map, idx, min_samples=5, eps=2.5
            )
            total += len(clusters)
            self.get_logger().info(
                f'[SERVER] [INFO] {name}: {len(clusters)} clusters'
            )

        max_c = float(np.max(self.mineral_map))
        active = self.mineral_map[self.mineral_map > 0.1]
        avg_c = float(np.mean(active)) if active.size > 0 else 0.0
        cov = float(active.size / self.mineral_map.size * 100)
        self.get_logger().info(
            f'[SERVER] [INFO] Clusters={total} max={max_c:.2f} '
            f'avg={avg_c:.2f} coverage={cov:.1f}%'
        )

    # ------------------------------------------------------------------ #
    #  MAP GENERATION                                                      #
    # ------------------------------------------------------------------ #

    def _generate_obstacles(self) -> None:
        self.obstacle_map = np.zeros((_MAP.HEIGHT, _MAP.WIDTH))

        # Border walls
        self.obstacle_map[0, :] = _MAP.OBSTACLE_VALUE
        self.obstacle_map[-1, :] = _MAP.OBSTACLE_VALUE
        self.obstacle_map[:, 0] = _MAP.OBSTACLE_VALUE
        self.obstacle_map[:, -1] = _MAP.OBSTACLE_VALUE

        if self.generator_type == 'advanced':
            geology = np.mean(self.mineral_map, axis=2)
            self.obstacle_map[geology > _MAP.ROCKY_THRESHOLD] = _MAP.OBSTACLE_VALUE
            for _ in range(_MAP.OBSTACLE_DENSITY_ADV):
                x = np.random.randint(5, _MAP.WIDTH - 5)
                y = np.random.randint(5, _MAP.HEIGHT - 5)
                if geology[y, x] < 0.3:
                    self._add_circular_obstacle(x, y, np.random.randint(2, 6))
        else:
            for _ in range(_MAP.OBSTACLE_DENSITY):
                x = np.random.randint(_MAP.SPAWN_MARGIN, _MAP.WIDTH - _MAP.SPAWN_MARGIN)
                y = np.random.randint(_MAP.SPAWN_MARGIN, _MAP.HEIGHT - _MAP.SPAWN_MARGIN)
                self._add_circular_obstacle(x, y, np.random.randint(3, 8))

    def _add_circular_obstacle(self, cx: int, cy: int, radius: int) -> None:
        for i in range(max(0, cy - radius), min(_MAP.HEIGHT, cy + radius + 1)):
            for j in range(max(0, cx - radius), min(_MAP.WIDTH, cx + radius + 1)):
                if np.sqrt((i - cy) ** 2 + (j - cx) ** 2) <= radius:
                    self.obstacle_map[i, j] = _MAP.OBSTACLE_VALUE

    def _get_valid_start_position(self) -> Tuple[int, int]:
        """Return a free start position (max 200 attempts, fallback to centre)."""
        m = _MAP.SPAWN_MARGIN
        for _ in range(200):
            x = np.random.randint(m, _MAP.WIDTH - m)
            y = np.random.randint(m, _MAP.HEIGHT - m)
            if self.obstacle_map[y, x] != 0:
                continue
            if (
                self.generator_type == 'advanced'
                and np.max(self.mineral_map[y, x, :]) >= _MAP.SPAWN_MAX_MINERAL
            ):
                continue
            return (x, y)

        self.get_logger().warning(
            '[SERVER] [WARN] _get_valid_start_position: fallback to centre'
        )
        return (_MAP.WIDTH // 2, _MAP.HEIGHT // 2)

    def _calculate_science_targets(self) -> np.ndarray:
        unexplored = 1.0 - self.exploration_map
        if self.generator_type == 'advanced':
            grad = np.zeros((_MAP.HEIGHT, _MAP.WIDTH))
            for ch in range(4):
                layer = self.mineral_map[:, :, ch]
                gx, gy = np.gradient(layer)
                grad += np.abs(gx) + np.abs(gy)
            return unexplored * (grad / 4.0)
        mineral_potential = np.max(self.mineral_map, axis=2)
        return unexplored * mineral_potential

    # ------------------------------------------------------------------ #
    #  EPISODE MANAGEMENT                                                  #
    # ------------------------------------------------------------------ #

    def _episode_reset_callback(self, msg: String) -> None:
        """Server receives an episode-end signal and regenerates the map."""
        self.episode_count += 1
        self._regenerate_map()

    def _regenerate_map(self) -> None:
        """Regenerate a new mineral map for the next episode.

        A new seed is drawn each episode so agents cannot memorise deposit
        positions and must learn a general exploration strategy.
        """
        new_seed = int(np.random.randint(0, 100_000))

        with self.lock:
            if self.generator_type == 'advanced':
                self.mineral_generator = AdvancedMineralGenerator(
                    _MAP.WIDTH, _MAP.HEIGHT, seed=new_seed
                )
                # BUG FIX: pass seed explicitly — without this, generate_geological_map()
                # was always re-initialising its RNG to seed=0, producing the same map.
                self.mineral_map = self.mineral_generator.generate_geological_map(
                    seed=new_seed
                )
                self.underground_layers = (
                    self.mineral_generator.generate_underground_layers(
                        self.mineral_map, num_layers=3
                    )
                )
            else:
                np.random.seed(new_seed)
                self.mineral_map = self.mineral_generator.generate_mineral_map()

            self._generate_obstacles()
            self.exploration_map = np.zeros((_MAP.HEIGHT, _MAP.WIDTH))

            for i in range(_MAP.NUM_ROBOTS):
                self.robot_positions[i] = self._get_valid_start_position()

        self.get_logger().info(
            f'[SERVER] [INFO] Map regenerated '
            f'(episode={self.episode_count} seed={new_seed})'
        )

    # ------------------------------------------------------------------ #
    #  ROBOT INTERACTION                                                   #
    # ------------------------------------------------------------------ #

    def _make_position_callback(self, robot_id: int):
        def callback(msg: Pose2D) -> None:
            with self.lock:
                self.robot_positions[robot_id] = (msg.x, msg.y)
                self._update_exploration_map(msg.x, msg.y)
            self.get_logger().debug(
                f'[SERVER] [DEBUG] Robot {robot_id}: ({msg.x:.1f}, {msg.y:.1f})'
            )
        return callback

    def _make_cleaning_callback(self, robot_id: int):
        def callback(msg: Float32MultiArray) -> None:
            if len(msg.data) >= 2:
                self._clean_area(int(msg.data[0]), int(msg.data[1]))
        return callback

    def _update_exploration_map(self, x: float, y: float) -> None:
        xi, yi = int(x), int(y)
        r = _MAP.EXPLORATION_RADIUS
        for i in range(max(0, yi - r), min(_MAP.HEIGHT, yi + r + 1)):
            for j in range(max(0, xi - r), min(_MAP.WIDTH, xi + r + 1)):
                if np.sqrt((i - yi) ** 2 + (j - xi) ** 2) <= r:
                    self.exploration_map[i, j] = min(
                        1.0,
                        self.exploration_map[i, j] + _MAP.EXPLORATION_INCREMENT,
                    )

    def _clean_area(self, x: int, y: int) -> None:
        """Detect minerals at the robot's position — the static map is unchanged."""
        if not (0 <= y < _MAP.HEIGHT and 0 <= x < _MAP.WIDTH):
            return
        with self.lock:
            mineral_at = self.mineral_map[y, x, :]
        if np.max(mineral_at) > _MAP.MINERAL_DETECTION_THRESHOLD:
            self.get_logger().debug(
                f'[SERVER] [DEBUG] Mineral at ({x},{y}): '
                f'max={np.max(mineral_at):.2f}'
            )

    # ------------------------------------------------------------------ #
    #  MAP PUBLISHERS                                                      #
    # ------------------------------------------------------------------ #

    def _publish_maps(self) -> None:
        # Copy arrays under lock; publish outside to avoid holding the lock
        # during expensive serialisation and network I/O.
        with self.lock:
            mineral_flat = self.mineral_map.flatten()
            obstacle_flat = self.obstacle_map.flatten().astype(np.int8)
            science_targets = self._calculate_science_targets()
            science_flat = science_targets.flatten()

        mineral_msg = Float32MultiArray()
        mineral_msg.layout.dim.append(
            self._make_array_dim("height", _MAP.HEIGHT, _MAP.HEIGHT * _MAP.WIDTH * 4)
        )
        mineral_msg.layout.dim.append(
            self._make_array_dim("width", _MAP.WIDTH, _MAP.WIDTH * 4)
        )
        mineral_msg.layout.dim.append(self._make_array_dim("channels", 4, 4))
        mineral_msg.data = mineral_flat.tolist()
        self.mineral_pub.publish(mineral_msg)

        obstacle_msg = OccupancyGrid()
        obstacle_msg.header.stamp = self.get_clock().now().to_msg()
        obstacle_msg.header.frame_id = _MAP.MAP_FRAME
        obstacle_msg.info.width = _MAP.WIDTH
        obstacle_msg.info.height = _MAP.HEIGHT
        obstacle_msg.info.resolution = _MAP.MAP_RESOLUTION
        obstacle_msg.data = obstacle_flat.tolist()
        self.obstacle_pub.publish(obstacle_msg)

        science_msg = Float32MultiArray()
        science_msg.layout.dim.append(
            self._make_array_dim("height", _MAP.HEIGHT, _MAP.HEIGHT * _MAP.WIDTH)
        )
        science_msg.layout.dim.append(
            self._make_array_dim("width", _MAP.WIDTH, _MAP.WIDTH)
        )
        science_msg.data = science_flat.tolist()
        self.science_pub.publish(science_msg)

        self.map_publish_count += 1
        if self.map_publish_count % 5 == 0:
            self.get_logger().debug('[SERVER] [DEBUG] Maps published')

    def _publish_underground_layers(self) -> None:
        if self.generator_type != 'advanced' or not self.underground_layers:
            return

        with self.lock:
            layer_flat = self.underground_layers[0].flatten()

        underground_msg = Float32MultiArray()
        underground_msg.layout.dim.append(
            self._make_array_dim("height", _MAP.HEIGHT, _MAP.HEIGHT * _MAP.WIDTH * 4)
        )
        underground_msg.layout.dim.append(
            self._make_array_dim("width", _MAP.WIDTH, _MAP.WIDTH * 4)
        )
        underground_msg.layout.dim.append(
            self._make_array_dim("channels", 4, 4)
        )
        underground_msg.data = layer_flat.tolist()
        self.underground_pub.publish(underground_msg)

        self.underground_publish_count += 1
        if self.underground_publish_count % 10 == 0:
            self.get_logger().debug('[SERVER] [DEBUG] Underground layer published')

    def _publish_status(self) -> None:
        """Publish a lightweight status string.

        IMPORTANT: this callback must never call detect_mineral_clusters() or
        any other long-running operation. The server uses a single-threaded
        executor: blocking here delays publish_maps() and starves the agents
        of fresh mineral data (they receive all-zero maps after reset).
        """
        try:
            with self.lock:
                coverage = float(np.mean(self.exploration_map)) * 100.0
                active = len(self.robot_positions)
                if self.generator_type == 'advanced':
                    mineral_stats = (
                        f"REE cells: {int(np.sum(self.mineral_map > 0.3)):,}"
                    )
                else:
                    mineral_stats = (
                        f"density: {float(np.mean(self.mineral_map)) * 100:.1f}%"
                    )

            text = (
                f"robots={active} coverage={coverage:.1f}% {mineral_stats}"
            )
            msg = String()
            msg.data = text
            self.status_pub.publish(msg)

        except Exception as exc:
            self.get_logger().error(f'[SERVER] [ERROR] _publish_status: {exc}')
            return

        self.status_counter += 1
        if self.status_counter % 6 == 0:
            self.get_logger().info(f'[SERVER] [INFO] Status: {text}')

    # ------------------------------------------------------------------ #
    #  UTILITY                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_array_dim(label: str, size: int, stride: int) -> MultiArrayDimension:
        dim = MultiArrayDimension()
        dim.label = label
        dim.size = size
        dim.stride = stride
        return dim


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = REEExplorationServer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('[SERVER] [INFO] Shutting down')
    except Exception as exc:
        import traceback
        if node:
            node.get_logger().error(f'[SERVER] [ERROR] {exc}')
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
