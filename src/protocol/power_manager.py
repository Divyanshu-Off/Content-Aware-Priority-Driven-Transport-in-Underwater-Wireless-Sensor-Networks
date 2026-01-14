"""Power Management and Duty Cycling."""
from typing import Dict, List
from enum import Enum

class PowerMode(Enum):
    """Power modes for nodes."""
    ACTIVE = 0
    IDLE = 1
    SLEEP = 2
    DEEP_SLEEP = 3

class PowerManager:
    """Manages power modes and duty cycling."""
    
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.current_mode = PowerMode.ACTIVE
        self.duty_cycle = 100.0
        self.mode_history = []
        self.power_consumption: Dict[PowerMode, float] = {
            PowerMode.ACTIVE: 10.0,
            PowerMode.IDLE: 2.0,
            PowerMode.SLEEP: 0.5,
            PowerMode.DEEP_SLEEP: 0.05
        }
    
    def set_power_mode(self, mode: PowerMode) -> bool:
        """Change power mode."""
        self.current_mode = mode
        self.mode_history.append(mode)
        return True
    
    def set_duty_cycle(self, percentage: float) -> bool:
        """Set duty cycle percentage."""
        if 0 <= percentage <= 100:
            self.duty_cycle = percentage
            return True
        return False
    
    def get_power_consumption(self) -> float:
        """Get current power consumption."""
        return self.power_consumption.get(self.current_mode, 0.0)
    
    def enable_adaptive_duty_cycle(self, energy_level: float) -> float:
        """Adaptively adjust duty cycle based on energy."""
        if energy_level > 80:
            self.duty_cycle = 100.0
        elif energy_level > 50:
            self.duty_cycle = 50.0
        elif energy_level > 20:
            self.duty_cycle = 20.0
        else:
            self.duty_cycle = 5.0
        return self.duty_cycle
