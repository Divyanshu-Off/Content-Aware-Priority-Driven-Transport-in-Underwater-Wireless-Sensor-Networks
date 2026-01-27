"""Main network simulator for UWSN."""

import numpy as np
from typing import Dict, List, Optional
from .environment import UnderwaterEnvironment
from .node_manager import NodeManager

class NetworkSimulator:
    """Main simulator class for UWSN network simulation."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Initialize environment
        self.environment = UnderwaterEnvironment(
            width=self.config['environment']['width'],
            height=self.config['environment']['height'],
            depth=self.config['environment']['depth'],
            temperature=self.config['environment']['temperature']
        )
        
        # Initialize node manager
        self.node_manager = NodeManager(
            num_nodes=self.config['network']['num_nodes'],
            environment=self.environment
        )
        
        self.current_time = 0.0
        self.is_initialized = False
        
    @staticmethod
    def _default_config() -> Dict:
        """Default configuration for simulator."""
        return {
            'environment': {
                'width': 1000.0,
                'height': 1000.0,
                'depth': 200.0,
                'temperature': 10.0
            },
            'network': {
                'num_nodes': 20,
                'deployment_type': 'random',
                'transmission_range': 250.0
            },
            'simulation': {
                'duration': 1000.0,
                'time_step': 1.0
            }
        }
    
    def initialize(self):
        """Initialize the simulation."""
        # Deploy nodes
        self.node_manager.deploy_nodes(
            deployment_type=self.config['network']['deployment_type']
        )
        
        self.is_initialized = True
        print(f"Simulation initialized with {self.node_manager.num_nodes} nodes")
        print(f"Network stats: {self.node_manager.get_network_stats()}")
        
    def get_simulation_status(self) -> Dict:
        """Get current simulation status."""
        return {
            'current_time': self.current_time,
            'is_initialized': self.is_initialized,
            'network_alive': self.node_manager.check_network_alive(),
            'active_nodes': len(self.node_manager.get_active_nodes())
        }
