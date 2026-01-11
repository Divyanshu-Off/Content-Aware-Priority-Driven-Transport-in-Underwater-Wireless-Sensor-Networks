"""Node Class Implementation for UWSN Priority-Driven Transport

Implements the Node class with priority-based queue management per Phase 1 Design (§4).

Reference: docs/PHASE_1_DESIGN.md
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from queue import PriorityQueue
from .packet import Packet


@dataclass
class Node:
    """UWSN Sensor Node with priority-driven queuing and energy management.
    
    Maintains three separate priority queues (Phase 1 §4) and tracks energy consumption.
    
    Attributes:
        node_id: Unique node identifier (0-49)
        x, y, z: Position in 3D space (meters)
        initial_energy: Starting energy in Joules (default 10000)
        transmission_range: Max transmission distance in meters (default 300)
        queue_capacity: Max packets per node (default 100)
    """
    
    node_id: int
    x: float
    y: float
    z: float
    initial_energy: float = 10000.0
    transmission_range: float = 300.0
    queue_capacity: int = 100
    
    # Internal state
    residual_energy: float = field(init=False)
    neighbor_list: List[int] = field(default_factory=list, init=False)
    queue_A: PriorityQueue = field(default_factory=PriorityQueue, init=False)  # Emergency
    queue_B: PriorityQueue = field(default_factory=PriorityQueue, init=False)  # Control
    queue_C: PriorityQueue = field(default_factory=PriorityQueue, init=False)  # Routine
    stats: Dict = field(default_factory=dict, init=False)
    is_alive: bool = field(default=True, init=False)
    
    def __post_init__(self):
        """Initialize energy and statistics after dataclass initialization."""
        self.residual_energy = self.initial_energy
        
        # Initialize statistics tracking per class
        self.stats = {
            'tx': {0: 0, 1: 0, 2: 0},        # packets transmitted by class
            'rx': {0: 0, 1: 0, 2: 0},        # packets received by class
            'drop': {0: 0, 1: 0, 2: 0},      # packets dropped by class
            'energy': {0: 0.0, 1: 0.0, 2: 0.0}  # energy consumed per class
        }
    
    def get_position(self) -> Tuple[float, float, float]:
        """Return node position as tuple."""
        return (self.x, self.y, self.z)
    
    def set_neighbor_list(self, neighbors: List[int]) -> None:
        """Update neighbor list (nodes within transmission range).
        
        Args:
            neighbors: List of neighbor node IDs
        """
        self.neighbor_list = neighbors
    
    def enqueue_packet(self, pkt: Packet) -> bool:
        """Add packet to appropriate priority queue.
        
        Packets are enqueued to class-specific queues (Phase 1 §4.1).
        If total queue size exceeds capacity, drop packets by priority:
        - First drop Class C packets
        - Then drop Class B packets
        - Only drop Class A as last resort
        
        Args:
            pkt: Packet to enqueue
            
        Returns:
            bool: True if packet was enqueued, False if dropped
        """
        # Check total queue size across all classes
        total_queued = self.queue_A.qsize() + self.queue_B.qsize() + self.queue_C.qsize()
        
        if total_queued >= self.queue_capacity:
            # Queue full - apply drop policy
            # Try to drop lowest priority packets first
            if not self.queue_C.empty():
                # Drop a Class C packet
                self.queue_C.get()
                self.stats['drop'][0] += 1
            elif not self.queue_B.empty():
                # Drop a Class B packet
                self.queue_B.get()
                self.stats['drop'][1] += 1
            else:
                # Critical: must drop Class A packet
                self.queue_A.get()
                self.stats['drop'][2] += 1
        
        # Select queue based on class
        if pkt.class_id == 2:  # Class A (emergency)
            # Use negative priority for max-heap behavior
            queue = self.queue_A
        elif pkt.class_id == 1:  # Class B (control)
            queue = self.queue_B
        else:  # Class C (routine)
            queue = self.queue_C
        
        # Enqueue with (negative_priority, packet_id, packet) for sorting
        queue.put((-pkt.priority_level, pkt.pkt_id, pkt))
        return True
    
    def dequeue_next_packet(self) -> Optional[Packet]:
        """Get next packet to transmit using strict priority scheduling (Phase 1 §4.1).
        
        Service order:
        1. Class A packets (emergency) - highest priority
        2. Class B packets (control) - if A queue empty
        3. Class C packets (routine) - if A and B queues empty
        
        Returns:
            Packet if available, None otherwise
        """
        # Try Class A first
        if not self.queue_A.empty():
            _, _, pkt = self.queue_A.get()
            return pkt
        
        # Then Class B
        if not self.queue_B.empty():
            _, _, pkt = self.queue_B.get()
            return pkt
        
        # Finally Class C
        if not self.queue_C.empty():
            _, _, pkt = self.queue_C.get()
            return pkt
        
        return None
    
    def get_queue_status(self) -> Dict[int, int]:
        """Get current queue sizes for each class.
        
        Returns:
            Dict with counts: {class_id: queue_size}
        """
        return {
            0: self.queue_C.qsize(),  # Class C
            1: self.queue_B.qsize(),  # Class B
            2: self.queue_A.qsize()   # Class A
        }
    
    def is_queue_empty(self) -> bool:
        """Check if all queues are empty.
        
        Returns:
            bool: True if no packets queued
        """
        return self.queue_A.empty() and self.queue_B.empty() and self.queue_C.empty()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "ALIVE" if self.is_alive else "DEAD"
        return (f"Node(id={self.node_id}, pos=({self.x:.0f},{self.y:.0f},{self.z:.0f}), "
                f"energy={self.residual_energy:.1f}J, status={status})")
