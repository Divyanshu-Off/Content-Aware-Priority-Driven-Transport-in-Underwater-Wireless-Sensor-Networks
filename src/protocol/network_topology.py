"""Network Topology Management."""
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class NetworkTopology:
    """Manages network topology information."""
    
    def __init__(self):
        self.neighbors: Dict[int, Set[int]] = defaultdict(set)
        self.edges: Dict[Tuple[int, int], float] = {}
        self.node_count = 0
    
    def add_node(self, node_id: int) -> bool:
        """Add a node to the network."""
        if node_id not in self.neighbors:
            self.neighbors[node_id] = set()
            self.node_count += 1
            return True
        return False
    
    def add_link(self, node1: int, node2: int, weight: float = 1.0) -> bool:
        """Add a bidirectional link between nodes."""
        if node1 not in self.neighbors:
            self.add_node(node1)
        if node2 not in self.neighbors:
            self.add_node(node2)
        
        self.neighbors[node1].add(node2)
        self.neighbors[node2].add(node1)
        
        edge_key = tuple(sorted([node1, node2]))
        self.edges[edge_key] = weight
        return True
    
    def get_neighbors(self, node_id: int) -> Set[int]:
        """Get neighbors of a node."""
        return self.neighbors.get(node_id, set())
    
    def get_network_density(self) -> float:
        """Calculate network connectivity density."""
        if self.node_count <= 1:
            return 0.0
        max_edges = self.node_count * (self.node_count - 1) / 2
        return len(self.edges) / max_edges if max_edges > 0 else 0.0
