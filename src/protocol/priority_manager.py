"""Priority management system for UWSN packets."""
from typing import Optional, Dict

class PriorityManager:
    """Unified priority management for packets."""
    
    def __init__(self):
        self.priority_map: Dict = {}
        self.boost_history = []
        self.reclassifications = 0
    
    def set_priority(self, packet_id, priority_level: int) -> bool:
        """Set priority for a packet."""
        if 0 <= priority_level <= 3:
            self.priority_map[packet_id] = priority_level
            return True
        return False
    
    def get_priority(self, packet_id) -> Optional[int]:
        """Get current priority of packet."""
        return self.priority_map.get(packet_id)
    
    def apply_boost(self, packet_id) -> bool:
        """Apply priority boost to packet."""
        if packet_id in self.priority_map:
            current = self.priority_map[packet_id]
            if current < 3:
                self.priority_map[packet_id] = min(current + 1, 3)
                self.boost_history.append(packet_id)
                return True
        return False
    
    def get_statistics(self) -> Dict:
        """Return priority management statistics."""
        return {
            'total_packets': len(self.priority_map),
            'boosted_packets': len(set(self.boost_history)),
            'reclassifications': self.reclassifications
        }
