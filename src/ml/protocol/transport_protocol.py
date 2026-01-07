"""UWSN Priority-Driven Transport Protocol

Core transport layer implementation with ML-based prioritization.
"""

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PacketPriority(Enum):
    """Priority levels for packets."""
    CRITICAL = 3
    HIGH = 2
    NORMAL = 1
    LOW = 0


@dataclass
class Packet:
    """Data packet in UWSN."""
    packet_id: int
    source_node: int
    destination_node: int
    payload_size: int
    priority: PacketPriority
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    is_transmitted: bool = False
    transmission_start: Optional[float] = None
    transmission_end: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    
    @property
    def transmission_delay(self) -> Optional[float]:
        """Calculate transmission delay."""
        if self.transmission_end and self.transmission_start:
            return self.transmission_end - self.transmission_start
        return None
    
    @property
    def age(self) -> float:
        """Get packet age in seconds."""
        return datetime.now().timestamp() - self.timestamp


class PriorityQueue:
    """Priority-based packet queue."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = {priority: deque() for priority in PacketPriority}
    
    def enqueue(self, packet: Packet) -> bool:
        if self.size() >= self.max_size:
            logger.warning(f"Queue full. Dropping packet {packet.packet_id}")
            return False
        self.queue[packet.priority].append(packet)
        return True
    
    def dequeue(self) -> Optional[Packet]:
        for priority in sorted(PacketPriority, key=lambda x: x.value, reverse=True):
            if self.queue[priority]:
                return self.queue[priority].popleft()
        return None
    
    def size(self) -> int:
        return sum(len(q) for q in self.queue.values())


class EnergyManager:
    """Energy management for nodes."""
    
    def __init__(self, initial_energy: float = 1000.0):
        self.initial_energy = initial_energy
        self.current_energy = initial_energy
        self.transmission_power = 2.0
    
    def calculate_transmission_cost(self, payload_size: int, distance: float) -> float:
        path_loss = 0.0001 * (distance ** 2)
        transmission_time = (payload_size * 8) / 13000
        return self.transmission_power * transmission_time * (1 + path_loss)
    
    def consume_energy(self, cost: float) -> bool:
        if self.current_energy >= cost:
            self.current_energy -= cost
            return True
        return False
    
    def get_energy_level(self) -> float:
        return (self.current_energy / self.initial_energy) * 100


class UWSNTransportProtocol:
    """Main transport protocol implementation."""
    
    def __init__(self, node_id: int, classifier=None):
        self.node_id = node_id
        self.classifier = classifier
        self.packet_queue = PriorityQueue(max_size=500)
        self.energy_manager = EnergyManager()
        self.packet_counter = 0
        self.neighbor_nodes = {}
        self.transmission_stats = {
            'total_sent': 0,
            'total_dropped': 0,
            'total_failed': 0
        }
    
    def set_neighbors(self, neighbors: Dict[int, float]):
        self.neighbor_nodes = neighbors
    
    def classify_packet(self, features: np.ndarray) -> PacketPriority:
        if self.classifier:
            try:
                prediction = self.classifier.predict([features])[0]
                return PacketPriority(prediction)
            except:
                return PacketPriority.NORMAL
        return PacketPriority.NORMAL
    
    def receive_packet(self, source: int, destination: int, 
                      payload_size: int, features: Optional[np.ndarray] = None) -> bool:
        self.packet_counter += 1
        priority = self.classify_packet(features) if features is not None else PacketPriority.NORMAL
        packet = Packet(self.packet_counter, source, destination, payload_size, priority)
        success = self.packet_queue.enqueue(packet)
        if not success:
            self.transmission_stats['total_dropped'] += 1
        return success
    
    def transmit_next_packet(self) -> Tuple[bool, Optional[Packet]]:
        if self.energy_manager.get_energy_level() < 10:
            return False, None
        
        packet = self.packet_queue.dequeue()
        if not packet:
            return True, None
        
        if packet.destination_node not in self.neighbor_nodes:
            return False, packet
        
        distance = self.neighbor_nodes[packet.destination_node]
        cost = self.energy_manager.calculate_transmission_cost(packet.payload_size, distance)
        
        if not self.energy_manager.consume_energy(cost):
            self.packet_queue.enqueue(packet)
            return False, None
        
        packet.is_transmitted = True
        packet.transmission_start = datetime.now().timestamp()
        packet.transmission_end = packet.transmission_start + (packet.payload_size * 8) / 13000
        self.transmission_stats['total_sent'] += 1
        
        return True, packet
    
    def get_statistics(self) -> Dict:
        return {
            'node_id': self.node_id,
            'queue_size': self.packet_queue.size(),
            'energy_level': self.energy_manager.get_energy_level(),
            'stats': self.transmission_stats
        }
