"""Network Statistics and Performance Monitoring."""
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class NetworkStats:
    """Tracks network performance metrics."""
    total_packets: int = 0
    delivered_packets: int = 0
    lost_packets: int = 0
    total_hops: int = 0
    energy_consumed: float = 0.0
    simulation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class NetworkMonitor:
    """Monitors overall network statistics."""
    
    def __init__(self):
        self.stats = NetworkStats()
        self.node_stats: Dict[int, Dict] = {}
    
    def record_packet_sent(self, node_id: int) -> None:
        """Record packet transmission."""
        self.stats.total_packets += 1
        if node_id not in self.node_stats:
            self.node_stats[node_id] = {'sent': 0, 'delivered': 0}
        self.node_stats[node_id]['sent'] += 1
    
    def record_packet_delivered(self, node_id: int) -> None:
        """Record successful delivery."""
        self.stats.delivered_packets += 1
        if node_id in self.node_stats:
            self.node_stats[node_id]['delivered'] += 1
    
    def record_energy_used(self, energy: float) -> None:
        """Record energy consumption."""
        self.stats.energy_consumed += energy
    
    def get_delivery_ratio(self) -> float:
        """Calculate packet delivery ratio."""
        if self.stats.total_packets == 0:
            return 0.0
        return (self.stats.delivered_packets / self.stats.total_packets) * 100
    
    def get_average_hops(self) -> float:
        """Calculate average hops per packet."""
        if self.stats.delivered_packets == 0:
            return 0.0
        return self.stats.total_hops / self.stats.delivered_packets
