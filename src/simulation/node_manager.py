"""Node management for UWSN simulation."""

import numpy as np
from typing import List, Dict, Tuple, Optional
import random

class SensorNode:
    """Represents a sensor node in UWSN."""
    
    def __init__(self, node_id: int, position: Tuple[float, float, float],
                 initial_energy: float = 100.0, transmission_range: float = 250.0):
        self.node_id = node_id
        self.position = position
        self.energy = initial_energy
        self.max_energy = initial_energy
        self.transmission_range = transmission_range
        self.is_active = True
        self.neighbors = []
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_forwarded = 0
        self.routing_table = {}
        
    def consume_energy(self, amount: float):
        """Consume energy for transmission or reception."""
        self.energy = max(0, self.energy - amount)
        if self.energy == 0:
            self.is_active = False
            
    def get_remaining_energy_ratio(self) -> float:
        """Get remaining energy as a ratio."""
        return self.energy / self.max_energy if self.max_energy > 0 else 0
    
    def __repr__(self):
        return f"Node({self.node_id}, pos={self.position}, energy={self.energy:.2f})"


class NodeManager:
    """Manages all sensor nodes in the network."""
    
    def __init__(self, num_nodes: int = 20, environment=None):
        self.num_nodes = num_nodes
        self.environment = environment
        self.nodes = []
        self.sink_node = None
        
    def deploy_nodes(self, deployment_type: str = 'random'):
        """Deploy sensor nodes in the environment."""
        if deployment_type == 'random':
            self._deploy_random()
        elif deployment_type == 'grid':
            self._deploy_grid()
        elif deployment_type == 'clustered':
            self._deploy_clustered()
        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")
            
        # Set up sink node (last node)
        self.sink_node = self.nodes[-1]
        
        # Discover neighbors for all nodes
        self._discover_neighbors()
        
    def _deploy_random(self):
        """Random deployment of nodes."""
        for i in range(self.num_nodes):
            position = self.environment.get_random_position()
            node = SensorNode(i, position)
            self.nodes.append(node)
            
    def _deploy_grid(self):
        """Grid-based deployment of nodes."""
        grid_size = int(np.ceil(np.sqrt(self.num_nodes)))
        x_step = self.environment.width / grid_size
        y_step = self.environment.height / grid_size
        
        node_id = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if node_id >= self.num_nodes:
                    break
                    
                x = (i + 0.5) * x_step
                y = (j + 0.5) * y_step
                z = random.uniform(0, self.environment.depth)
                
                node = SensorNode(node_id, (x, y, z))
                self.nodes.append(node)
                node_id += 1
                
    def _deploy_clustered(self):
        """Clustered deployment of nodes."""
        num_clusters = max(3, self.num_nodes // 5)
        cluster_centers = []
        
        # Generate cluster centers
        for _ in range(num_clusters):
            center = self.environment.get_random_position()
            cluster_centers.append(center)
        
        # Deploy nodes around cluster centers
        for i in range(self.num_nodes):
            cluster_center = random.choice(cluster_centers)
            
            # Add random offset
            offset = (
                random.gauss(0, 50),
                random.gauss(0, 50),
                random.gauss(0, 20)
            )
            
            position = tuple(c + o for c, o in zip(cluster_center, offset))
            node = SensorNode(i, position)
            self.nodes.append(node)
            
    def _discover_neighbors(self):
        """Discover neighbors for all nodes."""
        for node in self.nodes:
            node.neighbors = []
            for other_node in self.nodes:
                if node.node_id != other_node.node_id:
                    if self.environment.is_in_range(
                        node.position, 
                        other_node.position, 
                        node.transmission_range
                    ):
                        node.neighbors.append(other_node.node_id)
                        
    def get_active_nodes(self) -> List[SensorNode]:
        """Get list of active nodes."""
        return [node for node in self.nodes if node.is_active]
    
    def get_node(self, node_id: int) -> Optional[SensorNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    def get_network_stats(self) -> Dict:
        """Get network statistics."""
        active_nodes = self.get_active_nodes()
        total_energy = sum(node.energy for node in self.nodes)
        avg_energy = total_energy / len(self.nodes) if self.nodes else 0
        
        total_neighbors = sum(len(node.neighbors) for node in active_nodes)
        avg_neighbors = total_neighbors / len(active_nodes) if active_nodes else 0
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'dead_nodes': len(self.nodes) - len(active_nodes),
            'total_energy': total_energy,
            'avg_energy': avg_energy,
            'avg_neighbors': avg_neighbors,
            'network_lifetime': len(active_nodes) / len(self.nodes) if self.nodes else 0
        }
    
    def update_node_energy(self, node_id: int, energy_consumed: float):
        """Update energy for a specific node."""
        node = self.get_node(node_id)
        if node:
            node.consume_energy(energy_consumed)
            
    def check_network_alive(self) -> bool:
        """Check if network is still operational."""
        active_count = len(self.get_active_nodes())
        # Network is alive if at least 50% of nodes are active
        return active_count >= (self.num_nodes * 0.5)
