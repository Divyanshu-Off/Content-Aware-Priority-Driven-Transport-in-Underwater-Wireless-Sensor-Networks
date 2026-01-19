"""Training environment for reinforcement learning routing agents."""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, List

class UWSNRoutingEnv(gym.Env):
    """OpenAI Gym environment for UWSN routing training."""
    
    def __init__(self, num_nodes: int = 20, max_hops: int = 10):
        super(UWSNRoutingEnv, self).__init__()
        
        self.num_nodes = num_nodes
        self.max_hops = max_hops
        
        # State: [current_node, dest_node, energy, delay, priority, hop_count]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([num_nodes, num_nodes, 1, 100, 3, max_hops]),
            dtype=np.float32
        )
        
        # Action: choose next hop node
        self.action_space = spaces.Discrete(num_nodes)
        
        self.current_node = 0
        self.dest_node = 0
        self.energy = 1.0
        self.delay = 0.0
        self.priority = 1
        self.hop_count = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_node = np.random.randint(0, self.num_nodes)
        self.dest_node = np.random.randint(0, self.num_nodes)
        while self.dest_node == self.current_node:
            self.dest_node = np.random.randint(0, self.num_nodes)
        
        self.energy = 1.0
        self.delay = 0.0
        self.priority = np.random.randint(1, 4)
        self.hop_count = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        # Update state based on action
        next_node = action
        
        # Calculate energy consumption
        distance = np.random.uniform(10, 100)  # Simulated distance
        energy_consumed = self._calculate_energy(distance)
        self.energy -= energy_consumed
        
        # Calculate delay
        hop_delay = distance / 1500  # Acoustic speed
        self.delay += hop_delay
        
        # Update position
        self.current_node = next_node
        self.hop_count += 1
        
        # Check termination conditions
        done = False
        reward = 0.0
        
        if self.current_node == self.dest_node:
            # Successful delivery
            reward = 100.0 - self.delay - (1.0 - self.energy) * 50
            done = True
        elif self.energy <= 0:
            # Out of energy
            reward = -100.0
            done = True
        elif self.hop_count >= self.max_hops:
            # Max hops reached
            reward = -50.0
            done = True
        else:
            # Intermediate reward
            reward = -self.delay * 0.1 - energy_consumed * 10
        
        info = {
            'energy': self.energy,
            'delay': self.delay,
            'hops': self.hop_count,
            'delivered': self.current_node == self.dest_node
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        return np.array([
            self.current_node,
            self.dest_node,
            self.energy,
            self.delay,
            self.priority,
            self.hop_count
        ], dtype=np.float32)
    
    def _calculate_energy(self, distance: float) -> float:
        """Calculate energy consumption for transmission."""
        # Simplified acoustic energy model
        return (distance ** 2) * 1e-6
