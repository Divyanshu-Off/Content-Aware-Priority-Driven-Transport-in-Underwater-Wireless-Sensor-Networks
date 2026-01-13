"""Congestion Control Mechanism for UWSN Transport."""
from typing import Dict, List, Optional
from enum import Enum
import time

class CongestionLevel(Enum):
    """Network congestion levels."""
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3

class CongestionController:
    """Manages congestion detection and mitigation."""
    
    def __init__(self):
        self.congestion_level = CongestionLevel.LOW
        self.packet_drop_rate = 0.0
        self.buffer_occupancy = {}
        self.threshold_high = 0.8
        self.threshold_critical = 0.95
        self.backoff_time = 0
    
    def detect_congestion(self, buffers: Dict[int, float]) -> CongestionLevel:
        """Detect congestion based on buffer occupancy."""
        avg_occupancy = sum(buffers.values()) / len(buffers) if buffers else 0.0
        
        if avg_occupancy >= self.threshold_critical:
            return CongestionLevel.CRITICAL
        elif avg_occupancy >= self.threshold_high:
            return CongestionLevel.HIGH
        elif avg_occupancy >= 0.5:
            return CongestionLevel.MODERATE
        else:
            return CongestionLevel.LOW
    
    def apply_backpressure(self, congestion_level: CongestionLevel) -> float:
        """Apply backpressure/backoff based on congestion."""
        backoff_factors = {
            CongestionLevel.LOW: 0.0,
            CongestionLevel.MODERATE: 0.1,
            CongestionLevel.HIGH: 0.5,
            CongestionLevel.CRITICAL: 1.0
        }
        return backoff_factors.get(congestion_level, 0.0)
    
    def adjust_transmission_rate(self, current_rate: float, 
                                 congestion_level: CongestionLevel) -> float:
        """Adjust transmission rate based on congestion."""
        reduction_factors = {
            CongestionLevel.LOW: 1.0,
            CongestionLevel.MODERATE: 0.75,
            CongestionLevel.HIGH: 0.5,
            CongestionLevel.CRITICAL: 0.25
        }
        factor = reduction_factors.get(congestion_level, 1.0)
        return current_rate * factor
