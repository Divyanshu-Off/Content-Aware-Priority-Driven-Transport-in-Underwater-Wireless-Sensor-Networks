"""Energy Optimization for UWSN Nodes."""
from typing import Dict, List, Tuple
import math

class EnergyOptimizer:
    """Optimizes energy consumption in nodes."""
    
    def __init__(self, initial_energy: float = 100.0):
        self.initial_energy = initial_energy
        self.current_energy = initial_energy
        self.energy_consumed = 0.0
        self.transmission_cost = 0.5
        self.idle_cost = 0.01
        self.sleep_mode_enabled = False
    
    def calculate_transmission_energy(self, packet_size: int, 
                                     distance: float) -> float:
        """Calculate energy needed for transmission."""
        # Energy = base_cost + (size * distance_factor)
        distance_factor = math.log(distance + 1)
        energy_needed = self.transmission_cost * packet_size * distance_factor
        return energy_needed
    
    def transmit_packet(self, packet_size: int, distance: float) -> bool:
        """Attempt to transmit packet with available energy."""
        energy_needed = self.calculate_transmission_energy(packet_size, distance)
        
        if self.current_energy >= energy_needed:
            self.current_energy -= energy_needed
            self.energy_consumed += energy_needed
            return True
        return False
    
    def enable_sleep_mode(self) -> float:
        """Enable low-power sleep mode."""
        self.sleep_mode_enabled = True
        return self.idle_cost * 0.1
    
    def get_energy_percentage(self) -> float:
        """Get remaining energy as percentage."""
        return (self.current_energy / self.initial_energy) * 100
    
    def is_alive(self) -> bool:
        """Check if node has enough energy."""
        return self.current_energy > 0
