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

    
    def run_simulation(self, duration: Optional[float] = None):
        """Run the simulation for specified duration."""
        if not self.is_initialized:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")
        
        sim_duration = duration or self.config['simulation']['duration']
        time_step = self.config['simulation']['time_step']
        
        print(f"\nStarting simulation for {sim_duration} seconds...")
        
        while self.current_time < sim_duration:
            # Update environment
            self.environment.update_environment(time_step)
            
            # Check if network is still alive
            if not self.node_manager.check_network_alive():
                print(f"Network died at time {self.current_time}")
                break
            
            # Advance time
            self.current_time += time_step
            
            # Print progress every 100 time units
            if int(self.current_time) % 100 == 0:
                stats = self.node_manager.get_network_stats()
                print(f"Time: {self.current_time:.1f}s - Active nodes: {stats['active_nodes']}")
        
        print(f"\nSimulation completed at time {self.current_time}")
        self._print_final_statistics()
    
    def _print_final_statistics(self):
        """Print final simulation statistics."""
        stats = self.node_manager.get_network_stats()
        
        print("\n" + "="*50)
        print("SIMULATION STATISTICS")
        print("="*50)
        print(f"Total simulation time: {self.current_time:.2f} seconds")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Active nodes: {stats['active_nodes']}")
        print(f"Dead nodes: {stats['dead_nodes']}")
        print(f"Average energy: {stats['avg_energy']:.2f}")
        print(f"Network lifetime: {stats['network_lifetime']:.2%}")
        print("="*50)

    
    def step(self, time_step: float = 1.0):
        """Execute one simulation step."""
        if not self.is_initialized:
            raise RuntimeError("Simulation not initialized")
        
        self.environment.update_environment(time_step)
        self.current_time += time_step
        
        return self.get_simulation_status()
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.current_time = 0.0
        self.is_initialized = False
        
        # Reinitialize environment and nodes
        self.environment = UnderwaterEnvironment(
            width=self.config['environment']['width'],
            height=self.config['environment']['height'],
            depth=self.config['environment']['depth'],
            temperature=self.config['environment']['temperature']
        )
        
        self.node_manager = NodeManager(
            num_nodes=self.config['network']['num_nodes'],
            environment=self.environment
        )
        
        print("Simulation reset to initial state")
    
    def save_configuration(self, filepath: str):
        """Save simulation configuration to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved to {filepath}")
    
    @staticmethod
    def load_configuration(filepath: str) -> Dict:
        """Load simulation configuration from file."""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {filepath}")
        return config
