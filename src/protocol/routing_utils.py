"""Routing Utilities for UWSN."""
from typing import List, Dict, Set, Optional, Tuple
import heapq

class RoutingUtils:
    """Utility functions for routing operations."""
    
    @staticmethod
    def dijkstra(graph: Dict[int, List[Tuple[int, float]]], 
                 source: int, target: int) -> Optional[Tuple[List[int], float]]:
        """Find shortest path using Dijkstra's algorithm."""
        distances = {node: float('inf') for node in graph}
        distances[source] = 0
        previous = {}
        pq = [(0, source)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current == target:
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(source)
                return list(reversed(path)), distances[target]
            
            if current_dist > distances[current]:
                continue
            
            for neighbor, weight in graph.get(current, []):
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return None
    
    @staticmethod
    def calculate_path_cost(path: List[int], 
                           link_costs: Dict[Tuple[int, int], float]) -> float:
        """Calculate total cost of a path."""
        total = 0.0
        for i in range(len(path) - 1):
            edge = tuple(sorted([path[i], path[i+1]]))
            total += link_costs.get(edge, float('inf'))
        return total
