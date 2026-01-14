"""Link Quality Assessment and Management."""
from typing import Dict, List, Tuple
from collections import deque

class LinkQuality:
    """Manages link quality metrics."""
    
    def __init__(self, neighbor_id: int, window_size: int = 10):
        self.neighbor_id = neighbor_id
        self.window_size = window_size
        self.pkt_success_history = deque(maxlen=window_size)
        self.rssi_readings = deque(maxlen=window_size)
        self.reliability = 0.0
        self.link_cost = float('inf')
    
    def update_packet_result(self, success: bool, rssi: float) -> None:
        """Record packet transmission result."""
        self.pkt_success_history.append(1 if success else 0)
        self.rssi_readings.append(rssi)
        self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update link quality metrics."""
        if len(self.pkt_success_history) > 0:
            self.reliability = sum(self.pkt_success_history) / len(self.pkt_success_history)
        
        if len(self.rssi_readings) > 0:
            avg_rssi = sum(self.rssi_readings) / len(self.rssi_readings)
            self.link_cost = (100 - self.reliability * 100) + abs(avg_rssi)
    
    def get_reliability(self) -> float:
        """Get link reliability percentage."""
        return self.reliability * 100 if self.reliability else 0.0
    
    def is_stable(self, threshold: float = 0.8) -> bool:
        """Check if link is stable."""
        return self.reliability >= threshold
