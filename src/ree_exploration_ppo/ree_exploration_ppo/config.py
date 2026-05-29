from dataclasses import dataclass


@dataclass(frozen=True)
class TopicConfig:
    """Authoritative ROS2 topic names for the ree_exploration_ppo package.

    Per-robot topics use '{id}' as a placeholder — format at runtime with
    .format(id=robot_id).  Never hardcode topic names elsewhere.
    """

    # Published by ree_exploration_server (read-only for this package)
    MINERAL_MAP: str = "/mineral_map"
    OBSTACLE_MAP: str = "/obstacle_map"
    SCIENCE_TARGETS: str = "/science_targets"
    EPISODE_RESET: str = "/episode_reset"   # agent → server: triggers map regen

    # Agent → Trainer: JSON + base64 float16 encoded experience tuples
    # One topic per robot so each trainer only sees its own experiences.
    AGENT_EXPERIENCE_FMT: str = "/robot_{id}/ppo_experience"

    # Trainer → Agent: zlib-compressed weight state_dict (base64 String)
    WEIGHT_UPDATE_FMT: str = "/robot_{id}/ppo/weight_update"

    # Telemetry (agent outbound)
    AGENT_STEP: str = "/agent/step_completed"
    SHARED_DISCOVERIES: str = "/shared_discoveries"


@dataclass(frozen=True)
class NetworkConfig:
    """Architecture constants and PPO training hyperparameters.

    CNN follows the article architecture (Caccavale et al., 2023, Fig. 4):
        Conv(C, 32, 8×8/4) → Conv(32, 64, 4×4/2) → Conv(64, 64, 3×3/1)
        → Flatten → Dense(512) → Actor head (8) + Critic head (1)
    """

    # Map dimensions — fixed by ree_exploration_server; do not change
    MAP_HEIGHT: int = 100
    MAP_WIDTH: int = 100
    MINERAL_CHANNELS: int = 4

    # Observation: global 100×100 map — 6 canaux : 4 minéraux + obstacles + exploration
    OBS_CHANNELS: int = 6
    NUM_ACTIONS: int = 8      # N S W E NW NE SW SE

    # PPO core hyperparameters (Schulman et al. 2017)
    # LR: 3e-4 → 1e-4 → 3e-5 → 5e-5  (GradNorm stable at 0.6-0.9 → can afford faster updates)
    LEARNING_RATE: float = 5e-5
    GAMMA: float = 0.99               # discount factor
    # CLIP_EPS: 0.2 → 0.1  (ApproxKL rising to 9e-3 → approaching danger zone)
    CLIP_EPS: float = 0.1             # PPO clipping epsilon
    # N_EPOCHS: 10 → 4  (10 epochs overfits stale rollout → policy degrades)
    N_EPOCHS: int = 2
    MINI_BATCH_SIZE: int = 64         # mini-batch size for PPO updates
    GAE_LAMBDA: float = 0.95          # GAE smoothing parameter
    # VALUE_COEF: 0.5 (returns normalized in trainer → ValueLoss ≈ 1.0 → standard ratio)
    VALUE_COEF: float = 0.5
    # ENTROPY_COEF: 0.01 → 0.05  (entropy collapsed in ~10 updates)
    ENTROPY_COEF: float = 0.01
    ROLLOUT_STEPS: int = 512          # transitions collected before each update
    MAX_GRAD_NORM: float = 0.5        # gradient clipping norm

    # Episode termination
    MAX_STEPS_PER_EPISODE: int = 300
    MINERALS_TO_COMPLETE: int = 2
    CLEAN_THRESHOLD: float = 0.98

    # Thresholds
    MINERAL_DETECTION_THRESHOLD: float = 0.3
    OBSTACLE_THRESHOLD: int = 50
    # PENALTY: -1.0 → -0.1  (penalty/reward ratio was 1:20; now 1:200 → mineral
    # signal dominates as soon as one cell is found; sparse reward unlocked)
    PENALTY: float = -0.1
    # EXPLORATION_BONUS: reward for stepping on an unvisited cell (visited=0).
    # Counters the policy converging to repetitive paths → MineralRate decline.
    # Small enough not to dominate mineral reward (20.0 max) but > |PENALTY|.
    EXPLORATION_BONUS: float = 0.05
    # cpi in [0,4] → reward = cpi * REWARD_SCALE in [0, 20]  (was cpi*50 → [0,200])
    REWARD_SCALE: float = 5.0

    # Timing
    DECISION_HZ: float = 10.0
    CMD_VEL_LINEAR: float = 0.15

    # Spawning margin (cells from map border)
    SPAWN_MARGIN: int = 5

    # Weight broadcast interval (every N rollout updates)
    WEIGHT_BROADCAST_FREQ: int = 1    # broadcast after every rollout update

    # Paths (relative to CWD at launch — workspace root)
    MODEL_BASE_DIR: str = "models/ppo"
    TB_BASE_DIR: str = "tensorboard_logs/ppo"
    CSV_BASE_DIR: str = "logs/ppo"

    # Watchdog: warn if no experience arrives within this window (seconds)
    WATCHDOG_TIMEOUT_SEC: float = 30.0

    # KL early stopping: halt epoch if mean ApproxKL exceeds this threshold
    # (Schulman blog recommendation: 0.01–0.02; set conservatively at 0.015)
    TARGET_KL: float = 0.05
