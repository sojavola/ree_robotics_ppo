# REE Robotics PPO — Multi-Robot Rare Earth Exploration

Decentralized multi-robot exploration system using **Proximal Policy Optimization (PPO)** for Rare Earth Element (REE) detection on a 100×100 grid map. Implemented in **ROS2 (Humble)** with a shared CNN backbone (Actor-Critic), designed as a benchmark against QMIX and DQN algorithms.

---

## Architecture

### Paradigm: DTDE (Decentralized Training, Decentralized Execution)

Each robot has its own independent PPO agent and trainer. There is no centralized controller — robots learn independently from their own experience.

```
┌─────────────────────────────────────────────────────────┐
│                  ree_exploration_server                  │
│  - Generates 100×100 REE mineral maps (4 types)         │
│  - Publishes /mineral_map, /obstacle_map                 │
│  - Regenerates map every episode (300 steps)             │
└──────────────────┬──────────────────────────────────────┘
                   │ /mineral_map  /obstacle_map
       ┌───────────┼───────────────────────────┐
       ▼           ▼                           ▼
┌─────────────┐ ┌─────────────┐     ┌─────────────┐
│ ppo_agent_0 │ │ ppo_agent_1 │ ... │ ppo_agent_3 │
│  observe    │ │  observe    │     │  observe    │
│  decide     │ │  decide     │     │  decide     │
│  act        │ │  act        │     │  act        │
└──────┬──────┘ └──────┬──────┘     └──────┬──────┘
       │ /robot_0/     │ /robot_1/          │ /robot_3/
       │ ppo_experience│ ppo_experience     │ ppo_experience
       ▼               ▼                   ▼
┌───────────────┐ ┌───────────────┐   ┌───────────────┐
│ ppo_trainer_0 │ │ ppo_trainer_1 │...│ ppo_trainer_3 │
│  GAE          │ │  GAE          │   │  GAE          │
│  PPO-Clip     │ │  PPO-Clip     │   │  PPO-Clip     │
│  → weights    │ │  → weights    │   │  → weights    │
└───────────────┘ └───────────────┘   └───────────────┘
```

---

## Neural Network

**Shared CNN backbone** (Caccavale et al., 2023 architecture):

```
Input: (B, 6, 100, 100)
  Canal 0-3 : REE mineral concentrations (Oxides, Silicates, Phosphates, Carbonates)
  Canal 4   : Obstacle map
  Canal 5   : Visited cells (exploration map)

Conv(6→32,  8×8, stride=4)  → ReLU  →  (B, 32, 24, 24)
Conv(32→64, 4×4, stride=2)  → ReLU  →  (B, 64, 11, 11)
Conv(64→64, 3×3, stride=1)  → ReLU  →  (B, 64,  9,  9)
Flatten                               →  (B, 5184)
Linear(5184 → 512)           → ReLU  →  (B, 512)
                                          │
                    ┌─────────────────────┤
                    ▼                     ▼
             Actor head              Critic head
          Linear(512 → 8)          Linear(512 → 1)
          Categorical π(a|s)         V(s) value
          8 actions (N/S/E/W/NE/NW/SE/SW)
```

Initialization: orthogonal (actor gain=0.01, critic gain=1.0, backbone gain=√2).

---

## PPO Algorithm

Implementation following **Schulman et al. (2017)** with CleanRL/SB3 best practices:

| Component | Details |
|-----------|---------|
| Objective | PPO-Clip with ε=0.1 |
| Advantage | GAE-Lambda (γ=0.99, λ=0.95) |
| Value loss | Clipped MSE on normalized returns |
| Entropy | Bonus coefficient 0.01 |
| Rollout | 512 transitions per update |
| Epochs | 2 per rollout |
| Mini-batch | 64 transitions |
| Learning rate | 5e-5 (Adam) |
| Gradient clip | 0.5 |
| KL threshold | 0.05 (early stopping) |
| Bootstrap | V(s_{T+1}) for mid-episode rollouts |

---

## Reward System

Aligned with the QMIX benchmark for fair comparison:

| Event | Reward |
|-------|--------|
| First visit to mineral cell (conc ≥ 0.3) | `conc × 50 + conc × 30` |
| High concentration bonus (conc > 0.7) | `+30` |
| Revisit mineral cell | `conc × 50 × decay` (exponential) |
| New cell visited (coverage) | `+0.5` |
| Step penalty | `-0.05` |
| Collision with obstacle | `-5.0` |
| Loop detection (≥3 revisits in 10 steps) | `-0.5` |

Episodes are fixed at **300 steps** (identical to QMIX and DQN benchmarks).

---

## Project Structure

```
src/
├── ree_exploration_server/          # Map generation server
│   └── ree_exploration_server/
│       ├── server_node.py           # ROS2 node: generates and publishes maps
│       ├── advanced_mineral_generator.py  # Geological REE map generator
│       └── configs.py               # Server configuration
│
├── ree_exploration_ppo/             # PPO learning package
│   └── ree_exploration_ppo/
│       ├── ppo_agent_node.py        # Agent: observe → decide → act → publish
│       ├── ppo_trainer_node.py      # Trainer: GAE + PPO-Clip updates
│       ├── ppo_networks.py          # ActorCriticNetwork (CNN)
│       ├── rollout_buffer.py        # On-policy rollout buffer with GAE
│       ├── reward_system.py         # RealMineralRewardSystem (QMIX-aligned)
│       └── config.py                # All hyperparameters and topic names
│
└── ree_exploration_viz/             # Visualization
    ├── launch/
    │   ├── full_system.launch.py    # Launch everything (server + 4 agents + 4 trainers)
    │   └── visualization.launch.py # Launch RViz visualization only
    └── ree_exploration_viz/
        ├── visualization_node.py
        ├── mineral_heatmap_publisher.py
        └── robot_marker_publisher.py
```

---

## ROS2 Topics

| Topic | Direction | Content |
|-------|-----------|---------|
| `/mineral_map` | Server → Agents | Float32MultiArray (100×100×4) |
| `/obstacle_map` | Server → Agents | OccupancyGrid (100×100) |
| `/episode_reset` | Agent_0 → Server | Episode end signal (map regeneration) |
| `/robot_{id}/position` | Agent → * | Pose2D current position |
| `/robot_{id}/ppo_experience` | Agent → Trainer | JSON+base64 experience tuple |
| `/robot_{id}/ppo/weight_update` | Trainer → Agent | zlib-compressed state_dict |
| `/shared_discoveries` | Agent → * | High-reward mineral discoveries |

---

## Dependencies

- **ROS2 Humble** (Ubuntu 22.04)
- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0
- **NumPy**, **SciPy**
- `rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`

---

## Installation and Launch

```bash
# 1. Source ROS2
source /opt/ros/humble/setup.bash

# 2. Build all packages
cd src/
colcon build
source install/setup.bash

# 3. Launch full system (server + 4 robots + visualization)
ros2 launch ree_exploration_viz full_system.launch.py

# 4. Monitor training (separate terminal)
tensorboard --logdir src/tensorboard_logs/ppo/
```

---

## Continuous Learning

The system supports **continuous learning** across sessions:

- Checkpoints saved every 60 seconds and on SIGTERM
- On restart, weights, optimizer state, and episode/update counters are automatically restored
- Training resumes exactly where it stopped

```bash
# Checkpoints are saved at:
src/models/ppo/robot_{0,1,2,3}/latest.pt

# To start fresh (delete all training artifacts):
rm -rf src/models/ src/logs/ src/tensorboard_logs/
```

---

## Benchmark

This project implements PPO as one of three algorithms benchmarked on the same REE exploration task:

| Algorithm | Repository | Paradigm |
|-----------|-----------|---------|
| **PPO** (this repo) | `ree_robotics_ppo` | DTDE, on-policy |
| QMIX | `ree_robotics_qmix` | CTDE, off-policy |
| DQN | — | DTDE, off-policy |

All algorithms use identical: map size (100×100), episode length (300 steps), reward function, and 4 robots.

---

## Reference


> Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
