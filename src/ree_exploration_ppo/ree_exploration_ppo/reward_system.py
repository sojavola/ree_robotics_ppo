#!/usr/bin/env python3
"""
RealMineralRewardSystem — aligné avec QMIX reward_system.py pour benchmark.

Paramètres, logique et structure identiques au QMIX afin de garantir
une comparaison équitable entre les deux algorithmes.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from collections import deque


class RealMineralRewardSystem:
    """
    Système de récompenses basé sur les minéraux réels + coverage bonus.

    Identique au QMIX : mineral reward + coverage bonus + step penalty + anti-loop.
    """

    def __init__(self, grid_size=(100, 100), robot_id=0, coverage_weight=0.5):
        self.grid_size     = grid_size
        self.robot_id      = robot_id
        self.coverage_weight = coverage_weight

        self.reward_config = {
            # === RÉCOMPENSES MINÉRALES ===
            'mineral_base_reward':       50.0,
            'concentration_multiplier':  30.0,
            'high_concentration_bonus':  30.0,
            'mineral_threshold':          0.3,

            # === EXPLORATION ===
            'exploration_bonus': 1.0,
            'new_zone_bonus':    2.0,

            # === PÉNALITÉS ===
            'step_penalty':       -0.05,
            'collision_penalty':  -5.0,
            'revisiting_penalty': -0.5,

            # === EFFICACITÉ ===
            'efficiency_bonus': 0.3,

            # === Paramètres académiques ===
            'academic_penalty':       -2.0,
            'clean_threshold':         0.5,
            'gaussian_sigma':          0.9,
            'gaussian_update_freq':   15,
            'discount_factor':         0.99,
        }

        self.academic_heatmap = self._initialize_academic_heatmap()

        self.minerals_collected      = 0
        self.steps_without_mineral   = 0
        self.last_mineral_position   = None
        self.mineral_positions       = []
        self.mineral_positions_set   = set()
        self.concentration_history   = []

        self.visit_counts: dict      = {}
        self._recent_positions: deque = deque(maxlen=10)

        self._episode_visited: set   = set()

        self.visited_positions        = set()
        self.cleaned_positions        = set()
        self.unique_positions_count   = 0

        self.gaussian_step  = 0
        self.total_steps    = 0

        self.total_reward          = 0.0
        self.academic_reward_total = 0.0
        self.real_reward_total     = 0.0
        self.episode_rewards       = []

        self._ep_mineral:  float = 0.0
        self._ep_coverage: float = 0.0
        self._ep_penalty:  float = 0.0

    def _initialize_academic_heatmap(self):
        height, width = self.grid_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        for _ in range(6):
            x      = np.random.randint(15, width  - 15)
            y      = np.random.randint(15, height - 15)
            radius = np.random.randint(5, 12)
            for i in range(max(0, y - radius), min(height, y + radius)):
                for j in range(max(0, x - radius), min(width, x + radius)):
                    distance = np.sqrt((i - y) ** 2 + (j - x) ** 2)
                    if distance < radius:
                        concentration = 0.9 * (1.0 - distance / radius)
                        heatmap[i, j] = max(heatmap[i, j], concentration)
        heatmap = gaussian_filter(heatmap, sigma=1.5)
        return np.clip(heatmap, 0, 1)

    def calculate_reward(self, mineral_concentrations, position,
                         is_new_position=False, has_collision=False,
                         step_count=0, sensor_data=None):
        """
        Calcule la récompense : minérale + coverage bonus + pénalités.
        Identique à QMIX pour benchmark équitable.
        """
        x, y         = position
        position_key = (int(x), int(y))
        self.total_steps += 1

        _is_new_ep = position_key not in self._episode_visited

        real_reward     = self._calculate_real_mineral_reward(
            mineral_concentrations, position_key
        )
        strategic_bonus = self._calculate_strategic_bonus(position_key, step_count)

        reward = real_reward + strategic_bonus

        self._ep_mineral  += real_reward
        if _is_new_ep:
            self._ep_coverage += self.coverage_weight
        self._ep_penalty  += strategic_bonus - (
            self.coverage_weight if _is_new_ep else 0.0
        )

        self._update_tracking(position_key, is_new_position, real_reward > 0)
        self.total_reward      += reward
        self.real_reward_total += real_reward
        self.academic_reward_total = 0.0

        if np.isnan(reward) or np.isinf(reward):
            return 0.0

        return reward

    def _calculate_real_mineral_reward(self, concentrations, position_key):
        if not concentrations:
            return 0.0

        max_concentration = max(concentrations) if concentrations else 0.0

        if max_concentration < self.reward_config['mineral_threshold']:
            self.steps_without_mineral += 1
            return 0.0

        self.steps_without_mineral = 0

        if position_key in self.mineral_positions_set:
            visit_count   = self.visit_counts.get(position_key, 1)
            decay         = 0.05 / (1.0 + visit_count * 0.5)
            revisit_reward = (
                max_concentration
                * self.reward_config['mineral_base_reward']
                * decay
            )
            self.visit_counts[position_key] = visit_count + 1
            return revisit_reward

        mineral_reward  = max_concentration * self.reward_config['mineral_base_reward']
        mineral_reward += max_concentration * self.reward_config['concentration_multiplier']
        if max_concentration > 0.7:
            mineral_reward += self.reward_config['high_concentration_bonus']

        self.minerals_collected += 1
        self.mineral_positions.append(position_key)
        self.mineral_positions_set.add(position_key)
        self.visit_counts[position_key] = 1
        self.concentration_history.append(max_concentration)
        self.last_mineral_position = position_key

        return mineral_reward

    def _calculate_strategic_bonus(self, position_key, step_count):
        bonus = 0.0

        if position_key not in self._episode_visited:
            bonus += self.coverage_weight

        self._episode_visited.add(position_key)

        bonus += self.reward_config['step_penalty']

        self._recent_positions.append(position_key)
        loop_count = list(self._recent_positions).count(position_key)
        if loop_count >= 3:
            bonus -= 0.5

        return bonus

    def _update_tracking(self, position_key, is_new_position, found_mineral):
        if is_new_position and position_key not in self.visited_positions:
            self.visited_positions.add(position_key)
            self.unique_positions_count += 1

    def get_statistics(self):
        avg_concentration = (
            np.mean(self.concentration_history)
            if self.minerals_collected > 0 else 0.0
        )
        coverage          = (
            len(self.visited_positions)
            / (self.grid_size[0] * self.grid_size[1]) * 100
        )
        total_cells             = self.grid_size[0] * self.grid_size[1]
        current_priority_sum    = np.sum(self.academic_heatmap)
        cleanliness_percentage  = 100 * (1.0 - current_priority_sum / total_cells)

        return {
            'robot_id':                 self.robot_id,
            'minerals_collected':        self.minerals_collected,
            'total_reward':              self.total_reward,
            'academic_reward':           self.academic_reward_total,
            'real_reward':               self.real_reward_total,
            'visited_positions':         len(self.visited_positions),
            'episode_visited':           len(self._episode_visited),
            'coverage_percentage':       coverage,
            'cleanliness_percentage':    cleanliness_percentage,
            'avg_mineral_concentration': avg_concentration,
            'steps_without_mineral':     self.steps_without_mineral,
            'gaussian_updates':          (
                self.gaussian_step
                // self.reward_config['gaussian_update_freq']
            ),
            'heatmap_mean':   float(np.mean(self.academic_heatmap)),
            'heatmap_std':    float(np.std(self.academic_heatmap)),
            'bellman_discount': self.reward_config['discount_factor'],
        }

    def reset_episode(self):
        """
        Réinitialise pour un nouvel épisode.
        _episode_visited est remis à zéro (coverage repart à 0).
        visited_positions N'est PAS effacé (suivi global, identique à QMIX Fix A).
        """
        self.episode_rewards.append(self.total_reward)
        self.total_reward          = 0.0
        self.academic_reward_total = 0.0
        self.real_reward_total     = 0.0
        self.cleaned_positions.clear()
        self.mineral_positions_set.clear()
        self.minerals_collected     = 0
        self.mineral_positions      = []
        self.unique_positions_count = 0
        self.steps_without_mineral  = 0
        self.total_steps            = 0
        self.gaussian_step          = 0
        self.visit_counts.clear()
        self._recent_positions.clear()
        self._episode_visited.clear()
        self._ep_mineral  = 0.0
        self._ep_coverage = 0.0
        self._ep_penalty  = 0.0

    def get_episode_reward_components(self):
        return {
            'mineral':  round(self._ep_mineral,  3),
            'coverage': round(self._ep_coverage, 3),
            'penalty':  round(self._ep_penalty,  3),
        }
