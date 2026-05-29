from dataclasses import dataclass


@dataclass(frozen=True)
class ServerTopicConfig:
    """Centralised ROS2 topic names for the REE exploration server."""

    # Publishers
    MINERAL_MAP: str = "/mineral_map"
    OBSTACLE_MAP: str = "/obstacle_map"
    SCIENCE_TARGETS: str = "/science_targets"
    SYSTEM_STATUS: str = "/system_status"
    UNDERGROUND_LAYERS: str = "/underground_layers"

    # Subscriptions
    EPISODE_RESET: str = "/episode_reset"


@dataclass(frozen=True)
class MapConfig:
    """Environment geometry and physics parameters."""

    WIDTH: int = 100
    HEIGHT: int = 100
    NUM_ROBOTS: int = 4

    # Obstacle encoding (ROS OccupancyGrid convention: 0=free, 100=occupied)
    OBSTACLE_VALUE: int = 100
    OBSTACLE_DENSITY: int = 8        # random internal obstacles (basic generator)
    OBSTACLE_DENSITY_ADV: int = 8    # random internal obstacles (advanced generator)
    ROCKY_THRESHOLD: float = 0.7     # geology mean > this → rocky cell
    OBSTACLE_NEARBY_THRESHOLD: int = 80

    # Exploration
    EXPLORATION_RADIUS: int = 5
    EXPLORATION_INCREMENT: float = 0.3

    # Starting position
    SPAWN_MARGIN: int = 10
    SPAWN_MAX_MINERAL: float = 0.8   # reject start cells with mineral > this

    # Mineral detection threshold for clean_area()
    MINERAL_DETECTION_THRESHOLD: float = 0.3

    # OccupancyGrid metadata
    MAP_RESOLUTION: float = 0.1
    MAP_FRAME: str = "map"

    # Timer intervals (seconds)
    MAP_PUBLISH_PERIOD: float = 0.5
    STATUS_PUBLISH_PERIOD: float = 5.0
    UNDERGROUND_PUBLISH_PERIOD: float = 5.0
