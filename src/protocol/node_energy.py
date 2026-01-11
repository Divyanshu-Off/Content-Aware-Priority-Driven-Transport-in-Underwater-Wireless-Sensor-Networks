"""Energy Management Methods for Node Class

Extension methods for energy consumption, transmission modeling, and status tracking.
Based on Phase 1 Design (§6 Energy Model).

Reference: docs/PHASE_1_DESIGN.md
"""

from typing import Tuple


def consume_energy_transmission(node, packet, distance: float) -> float:
    """Calculate and deduct energy for packet transmission (Phase 1 §6).
    
    Energy formula:
        Energy_tx = (Payload_size + Overhead) * (Tx_power / Bit_rate) + Tx_power * Propagation_delay
        where:
            Overhead = 50 bytes
            Tx_power = 2 Watts
            Bit_rate = 1000 bps
            Propagation_delay = distance / 1500 seconds
    
    Example (Phase 1 §6): Transmitting 128-byte Class A packet over 200m:
        Energy = (128+50)*8/1000*2 + 2*(200/1500) ≈ 1.45 J
    
    Args:
        node: The transmitting node
        packet: Packet being transmitted
        distance: Distance to receiver in meters
        
    Returns:
        float: Energy consumed in Joules
    """
    OVERHEAD = 50  # bytes (Phase 1 §3.1)
    TX_POWER = 2.0  # Watts
    BIT_RATE = 1000.0  # bps (Phase 1 §1.2)
    SOUND_SPEED = 1500.0  # m/s in seawater
    
    total_bytes = OVERHEAD + packet.payload_size
    transmission_time = (total_bytes * 8) / BIT_RATE  # seconds
    propagation_delay = distance / SOUND_SPEED  # seconds
    
    energy_tx = (transmission_time * TX_POWER) + (propagation_delay * TX_POWER)
    
    # Deduct from node
    node.residual_energy -= energy_tx
    node.stats['energy'][packet.class_id] += energy_tx
    
    # Check if node is now dead
    if node.residual_energy <= 0:
        node.is_alive = False
        node.residual_energy = 0
    
    return energy_tx


def consume_energy_reception(node, packet) -> float:
    """Calculate and deduct energy for packet reception (Phase 1 §6).
    
    Energy formula:
        Energy_rx = Payload_size * (Rx_power / Bit_rate)
        where:
            Rx_power = 0.5 Watts (Phase 1 §1.2)
            Bit_rate = 1000 bps
    
    Args:
        node: The receiving node
        packet: Packet being received
        
    Returns:
        float: Energy consumed in Joules
    """
    RX_POWER = 0.5  # Watts
    BIT_RATE = 1000.0  # bps
    
    reception_time = (packet.payload_size * 8) / BIT_RATE  # seconds
    energy_rx = reception_time * RX_POWER
    
    node.residual_energy -= energy_rx
    
    return energy_rx


def consume_energy_idle(node, duration: float = 1.0) -> float:
    """Calculate energy consumed while idle (Phase 1 §6).
    
    Energy formula:
        Energy_idle = Idle_power * duration
        where:
            Idle_power = 0.1 Watts (Phase 1 §1.2)
    
    Args:
        node: The idle node
        duration: Duration in seconds (default 1.0)
        
    Returns:
        float: Energy consumed in Joules
    """
    IDLE_POWER = 0.1  # Watts
    energy_idle = IDLE_POWER * duration
    node.residual_energy -= energy_idle
    
    return energy_idle


def get_energy_level_percentage(node) -> float:
    """Get remaining energy as percentage of initial energy.
    
    Returns:
        float: Percentage from 0 to 100
    """
    percentage = (node.residual_energy / node.initial_energy) * 100
    return max(0, min(100, percentage))


def is_energy_low(node, threshold_percent: float = 20.0) -> bool:
    """Check if node energy is below threshold.
    
    Used for priority boosting (Phase 1 §5.1):
    Packets from low-energy nodes get priority boost.
    
    Args:
        node: The node to check
        threshold_percent: Energy threshold in percent (default 20%)
        
    Returns:
        bool: True if energy is below threshold
    """
    threshold_joules = (threshold_percent / 100.0) * node.initial_energy
    return node.residual_energy < threshold_joules


def get_energy_status_summary(node) -> dict:
    """Get comprehensive energy status for the node.
    
    Returns:
        dict: Energy summary with multiple metrics
    """
    percentage = get_energy_level_percentage(node)
    is_low = is_energy_low(node)
    
    return {
        'residual_joules': round(node.residual_energy, 2),
        'initial_joules': node.initial_energy,
        'percentage_remaining': round(percentage, 1),
        'is_low_energy': is_low,
        'is_alive': node.is_alive,
        'tx_energy_used': round(node.stats['energy'][0] + node.stats['energy'][1] + node.stats['energy'][2], 2)
    }
