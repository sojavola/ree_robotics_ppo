"""
Multi-robot launch file — ree_exploration_ppo.

Starts 4 independent (ppo_agent, ppo_trainer) pairs (robot_id 0–3).
Each pair is fully isolated: its own topics, model dir, TensorBoard dir, CSV dir
and rollout buffer — no shared weights, no shared memory (DTDE strict).

Usage:
    ros2 launch ree_exploration_ppo ree_ppo_multi.launch.py

Per-robot topics (example for robot_id=1):
    /robot_1/ppo_experience      — agent → trainer (experience tuples)
    /robot_1/ppo/weight_update   — trainer → agent (updated actor_critic weights)

Per-robot paths (example for robot_id=1):
    models/ppo/robot_1/latest.pt
    tensorboard_logs/ppo/robot_1/
    logs/ppo/robot_1/episodes.csv
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _robot_pair(robot_id: int, params_file: str) -> list:
    """Return [ppo_agent_node, ppo_trainer_node] for one robot."""
    rid = str(robot_id)
    agent = Node(
        package='ree_exploration_ppo',
        executable='ppo_agent',
        name=f'ppo_agent_{rid}',
        parameters=[
            params_file,
            {'robot_id': robot_id},
        ],
        output='screen',
        emulate_tty=True,
    )
    trainer = Node(
        package='ree_exploration_ppo',
        executable='ppo_trainer',
        name=f'ppo_trainer_{rid}',
        parameters=[
            params_file,
            {'robot_id': robot_id},
        ],
        output='screen',
        emulate_tty=True,
    )
    return [agent, trainer]


def generate_launch_description() -> LaunchDescription:
    pkg_share  = get_package_share_directory('ree_exploration_ppo')
    params_file = os.path.join(pkg_share, 'config', 'ppo_params.yaml')

    nodes = []
    for rid in range(4):
        nodes.extend(_robot_pair(rid, params_file))

    return LaunchDescription(nodes)
