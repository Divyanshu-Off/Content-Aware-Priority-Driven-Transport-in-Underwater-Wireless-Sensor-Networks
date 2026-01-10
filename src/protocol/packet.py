"""Packet Class Implementation for UWSN Priority-Driven Transport

Implements the Packet class with content-aware priority mapping per Phase 1 Design (§3, §5).

Reference: docs/PHASE_1_DESIGN.md
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Packet:
    """UWSN Packet with content-aware priority mapping.
    
    Attributes matching Phase 1 Packet Schema (§3):
        pkt_id: Unique packet identifier
        src_node_id: Source node ID (0-49)
        dst_node_id: Destination node ID (typically sink at 50)
        timestamp: Time packet was created (simulation time units)
        class_id: Traffic class (0=C/routine, 1=B/control, 2=A/emergency)
        payload_size: Payload size in bytes
        ttl: Time-to-live (max hops)
        hop_count: Current number of hops traveled
        depth: Node depth in meters
        residual_energy: Node's residual energy in Joules
        sensed_value: Raw sensor reading
        is_anomaly: Whether sensed value deviates from baseline
        priority_level: Computed priority (0-3), auto-calculated from class + context
    """
    
    pkt_id: int
    src_node_id: int
    dst_node_id: int
    timestamp: float
    class_id: int  # 0=C, 1=B, 2=A
    payload_size: int
    depth: float
    residual_energy: float
    sensed_value: float = 0.0
    is_anomaly: bool = False
    ttl: int = 10
    hop_count: int = 0
    priority_level: int = 0  # Will be computed
    
    def __post_init__(self):
        """Auto-compute priority level after initialization."""
        self.priority_level = self.compute_priority()
    
    def compute_priority(self) -> int:
        """Content-to-Priority Mapping (Phase 1 §5.1)
        
        Maps packet class and context (anomaly, energy) to priority level 0-3.
        
        Formula:
            priority_level = base_priority + context_boost
            where:
                base_priority = 2 if class_A, 1 if class_B, 0 if class_C
                context_boost = +1 if anomaly detected
                              + +1 if residual_energy < 20% (2000J of 10000J)
            result = min(base_priority + context_boost, 3)
        
        Returns:
            int: Priority level 0-3 (0=lowest, 3=highest)
        
        Examples (Phase 1 §5.2):
            Class C, no anomaly, sufficient energy → priority 0
            Class C, anomaly detected → priority 1
            Class B, no anomaly, sufficient energy → priority 1
            Class B, no anomaly, low energy → priority 2
            Class A, no anomaly, sufficient energy → priority 2
            Class A, anomaly detected, low energy → priority 3 (max)
        """
        # Base priority from class (Phase 1 §2.1)
        base_priority = {
            0: 0,  # Class C (routine) → 0
            1: 1,  # Class B (control) → 1
            2: 2,  # Class A (emergency) → 2
        }[self.class_id]
        
        # Context-based boost
        context_boost = 0
        
        # Boost 1: Anomaly detected
        if self.is_anomaly:
            context_boost += 1
        
        # Boost 2: Node near end-of-life (residual_energy < 20% of 10000J)
        INITIAL_ENERGY = 10000.0
        ENERGY_THRESHOLD = 0.2 * INITIAL_ENERGY  # 2000J
        if self.residual_energy < ENERGY_THRESHOLD:
            context_boost += 1
        
        # Cap at maximum priority level 3
        final_priority = min(base_priority + context_boost, 3)
        return final_priority
    
    def get_transmission_size(self) -> int:
        """Total bytes to transmit including overhead.
        
        Per Phase 1 §3.1:
            Overhead: 50 bytes (headers, metadata)
            Payload: 64-256 bytes (class-dependent)
            Total: 114-306 bytes
        
        Returns:
            int: Total transmission size in bytes
        """
        OVERHEAD = 50  # bytes
        return OVERHEAD + self.payload_size
    
    def update_hop_count(self) -> None:
        """Increment hop count when packet is forwarded."""
        self.hop_count += 1
    
    def is_expired(self) -> bool:
        """Check if packet has exceeded TTL.
        
        Returns:
            bool: True if hop_count >= ttl
        """
        return self.hop_count >= self.ttl
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        class_names = {0: 'C', 1: 'B', 2: 'A'}
        return (f"Packet(id={self.pkt_id}, src={self.src_node_id}, "
                f"class={class_names[self.class_id]}, "
                f"priority={self.priority_level}, "
                f"hops={self.hop_count}/{self.ttl})")
    
    def get_class_name(self) -> str:
        """Return human-readable class name.
        
        Returns:
            str: 'Routine', 'Control', or 'Emergency'
        """
        return {0: 'Routine', 1: 'Control', 2: 'Emergency'}[self.class_id]
    
    def get_priority_name(self) -> str:
        """Return human-readable priority name.
        
        Returns:
            str: 'Lowest', 'Low', 'High', or 'Critical'
        """
        return {0: 'Lowest', 1: 'Low', 2: 'High', 3: 'Critical'}[self.priority_level]
