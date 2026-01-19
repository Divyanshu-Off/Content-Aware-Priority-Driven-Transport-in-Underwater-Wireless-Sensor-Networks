"""Reward functions for reinforcement learning-based routing."""

import numpy as np
from typing import Dict, Tuple

class RLRewardCalculator:
    """Calculate rewards for RL routing decisions."""
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        self.alpha = alpha  # Energy weight
        self.beta = beta    # Delay weight
        self.gamma = gamma  # Success weight
        
    def calculate_reward(self, state: Dict, action: int, next_state: Dict) -> float:
        """Calculate reward based on transition."""
        energy_reward = self._energy_reward(state, next_state)
        delay_reward = self._delay_reward(state, next_state)
        success_reward = self._success_reward(next_state)
        
        total_reward = (self.alpha * energy_reward + 
                       self.beta * delay_reward + 
                       self.gamma * success_reward)
        
        return total_reward
    
    def _energy_reward(self, state: Dict, next_state: Dict) -> float:
        """Reward for energy efficiency."""
        energy_consumed = state['energy'] - next_state['energy']
        # Normalize and invert (lower consumption = higher reward)
        return -energy_consumed / 100.0
    
    def _delay_reward(self, state: Dict, next_state: Dict) -> float:
        """Reward for low delay."""
        delay = next_state.get('delay', 0)
        # Exponential penalty for high delay
        return -np.exp(delay / 10.0)
    
    def _success_reward(self, next_state: Dict) -> float:
        """Reward for successful delivery."""
        if next_state.get('delivered', False):
            return 100.0
        elif next_state.get('dropped', False):
            return -50.0
        return 0.0
    
    def shaped_reward(self, state: Dict, action: int, next_state: Dict,
                     potential_function: callable) -> float:
        """Reward shaping using potential function."""
        base_reward = self.calculate_reward(state, action, next_state)
        shaping = potential_function(next_state) - potential_function(state)
        return base_reward + shaping
