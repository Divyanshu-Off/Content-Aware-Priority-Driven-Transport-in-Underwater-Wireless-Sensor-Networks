"""Transport Protocol for UWSN with Priority-based Scheduling."""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

class TransportState(Enum):
    """States of transport protocol."""
    IDLE = 0
    TRANSMITTING = 1
    RECEIVING = 2
    COLLISION_DETECTED = 3

@dataclass
class Route:
    """Represents a routing path."""
    path: List[int]
    hop_count: int
    reliability: float

class TransportProtocol:
    """Priority-driven transport protocol with scheduling."""
    
    def __init__(self, node_id: int, max_queue_size: int = 100):
        self.node_id = node_id
        self.state = TransportState.IDLE
        self.packet_queue: List = []
        self.max_queue_size = max_queue_size
        self.routes: Dict[int, Route] = {}
        self.transmission_count = 0
        self.collision_count = 0
    
    def schedule_packet(self, pkt) -> bool:
        """Schedule packet transmission based on priority."""
        if len(self.packet_queue) >= self.max_queue_size:
            return False
        
        # Insert packet in priority order (higher priority first)
        priority = getattr(pkt, 'priority', 1)
        insert_pos = len(self.packet_queue)
        
        for i, queued_pkt in enumerate(self.packet_queue):
            if priority > getattr(queued_pkt, 'priority', 1):
                insert_pos = i
                break
        
        self.packet_queue.insert(insert_pos, pkt)
        return True
    
    def select_route(self, dest_id: int, available_routes: List[Route]) -> Optional[Route]:
        """Select best route based on reliability and hop count."""
        if not available_routes:
            return None
        
        # Prefer routes with high reliability and low hop count
        best_route = max(available_routes, 
                        key=lambda r: r.reliability / (r.hop_count + 1))
        return best_route
