"""Simulation Helper Functions."""
from typing import List, Dict, Any
import random
import time

class SimulationHelpers:
    """Helper functions for simulation setup and execution."""
    
    @staticmethod
    def generate_random_topology(num_nodes: int, 
                                max_distance: float = 100.0) -> Dict:
        """Generate random network topology."""
        topology = {'nodes': {}, 'links': []}
        
        for i in range(num_nodes):
            topology['nodes'][i] = {
                'x': random.uniform(0, max_distance),
                'y': random.uniform(0, max_distance),
                'energy': 100.0
            }
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                x1, y1 = topology['nodes'][i]['x'], topology['nodes'][i]['y']
                x2, y2 = topology['nodes'][j]['x'], topology['nodes'][j]['y']
                distance = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
                
                if distance < max_distance * 0.4:
                    topology['links'].append({
                        'source': i,
                        'target': j,
                        'distance': distance
                    })
        
        return topology
    
    @staticmethod
    def create_test_packets(count: int, priority_distribution: Dict) -> List:
        """Create test packets with priority distribution."""
        packets = []
        priorities = list(priority_distribution.keys())
        weights = list(priority_distribution.values())
        
        for i in range(count):
            priority = random.choices(priorities, weights=weights)[0]
            packets.append({
                'id': i,
                'priority': priority,
                'size': random.randint(50, 500),
                'timestamp': time.time()
            })
        
        return packets
