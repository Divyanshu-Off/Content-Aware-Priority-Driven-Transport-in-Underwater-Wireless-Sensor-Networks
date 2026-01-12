"""Priority Classifier for UWSN Packets

Basic classifier interface for dynamic priority re-computation.
Based on Phase 1 Design (§5 Content-to-Priority Mapping).

Reference: docs/PHASE_1_DESIGN.md
"""

from typing import Optional
from .packet import Packet


class PriorityClassifier:
    """Static classifier for packet priority levels.
    
    This class provides methods to classify or re-classify packets
    based on their content and context, implementing the priority
    mapping rules from Phase 1 §5.
    """
    
    @staticmethod
    def classify(pkt: Packet) -> int:
        """Classify packet and return its priority level (0-3).
        
        This is a wrapper around the Packet.compute_priority() method,
        providing a standard interface for priority classification.
        
        Args:
            pkt: Packet to classify
            
        Returns:
            int: Priority level 0-3 (0=lowest, 3=highest)
        """
        return pkt.compute_priority()
    
    @staticmethod
    def reclassify(pkt: Packet) -> int:
        """Re-compute priority for an existing packet.
        
        Useful when packet context changes (e.g., node energy depletes)
        during transmission/queueing. Forces recomputation of priority.
        
        Args:
            pkt: Packet to reclassify
            
        Returns:
            int: New priority level 0-3
        """
        new_priority = pkt.compute_priority()
        pkt.priority_level = new_priority
        return new_priority
    
    @staticmethod
    def get_priority_name(priority_level: int) -> str:
        """Get human-readable name for priority level.
        
        Args:
            priority_level: Level 0-3
            
        Returns:
            str: Priority name
        """
        names = {0: 'Lowest', 1: 'Low', 2: 'High', 3: 'Critical'}
        return names.get(priority_level, 'Unknown')
    
    @staticmethod
    def should_boost_priority(pkt: Packet) -> bool:
        """Check if packet qualifies for priority boost.
        
        A packet qualifies for boost if:
        - It has anomalous sensor data, OR
        - Its source node is low on energy
        
        Args:
            pkt: Packet to check
            
        Returns:
            bool: True if packet should be boosted
        """
        # Anomaly check
        if pkt.is_anomaly:
            return True
        
        # Low energy check (< 20%)
        INITIAL_ENERGY = 10000.0
        ENERGY_THRESHOLD = 0.2 * INITIAL_ENERGY
        if pkt.residual_energy < ENERGY_THRESHOLD:
            return True
        
        return False
