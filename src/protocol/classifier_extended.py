"""Extended Priority Classifier with advanced methods."""
from typing import Dict, List, Tuple

class PriorityClassifierExtended:
    """Extended classifier with reclassification support."""
    
    def __init__(self):
        self.priority_names = {
            0: 'Lowest',
            1: 'Low',
            2: 'High',
            3: 'Critical'
        }
        self.reclassification_history = []
    
    def reclassify(self, pkt, new_priority: int) -> bool:
        """Reclassify packet to new priority level."""
        if new_priority not in self.priority_names:
            return False
        
        old_priority = getattr(pkt, 'priority', 1)
        self.reclassification_history.append({
            'packet_id': id(pkt),
            'old_priority': old_priority,
            'new_priority': new_priority
        })
        pkt.priority = new_priority
        return True
    
    def get_reclassification_count(self) -> int:
        """Get total reclassifications performed."""
        return len(self.reclassification_history)
