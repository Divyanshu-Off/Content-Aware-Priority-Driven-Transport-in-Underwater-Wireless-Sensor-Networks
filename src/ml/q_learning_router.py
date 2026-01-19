"""Q-Learning Based Adaptive Routing for UWSN."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

class QLearningRouter:
    """Q-Learning agent for adaptive routing decisions."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.1):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.episode_count = 0
        self.total_reward = 0
    
    def get_state(self, node_id: int, dest_id: int, energy: float, 
                  buffer_occupancy: float) -> Tuple:
        """Create state representation."""
        energy_level = int(energy / 20)
        buffer_level = int(buffer_occupancy / 0.2)
        return (node_id, dest_id, energy_level, buffer_level)
    
    def get_action(self, state: Tuple, available_neighbors: List[int]) -> int:
        """Select next hop using epsilon-greedy policy."""
        if not available_neighbors:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(available_neighbors)
        
        q_values = {neighbor: self.q_table[state][neighbor] 
                   for neighbor in available_neighbors}
        max_q = max(q_values.values()) if q_values else 0
        
        best_neighbors = [n for n, q in q_values.items() if q == max_q]
        return random.choice(best_neighbors)
    
    def calculate_reward(self, delivery_success: bool, hops: int, 
                        energy_used: float, delay: float) -> float:
        """Calculate reward for routing decision."""
Add QLearningRouter (Phase 3 Task 2 - Commit 1/3)        
        if delivery_success:
            reward += 100
            reward -= hops * 5
            reward -= energy_used * 2
            reward -= delay * 0.1
        else:
            reward -= 50
        
        return reward
    
    def update_q_value(self, state: Tuple, action: int, reward: float, 
                      next_state: Tuple, next_neighbors: List[int]) -> None:
        """Update Q-value using Q-learning update rule."""
        current_q = self.q_table[state][action]
        
        if next_neighbors:
            max_next_q = max([self.q_table[next_state][n] for n in next_neighbors])
        else:
            max_next_q = 0
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self, decay_rate: float = 0.995) -> None:
        """Decay exploration rate over time."""
        self.epsilon = max(0.01, self.epsilon * decay_rate)
    
    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table)
        }
