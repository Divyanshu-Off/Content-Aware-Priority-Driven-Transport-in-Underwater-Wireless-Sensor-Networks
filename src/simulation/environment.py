"""Underwater environment simulation for UWSN."""

import numpy as np
from typing import Tuple, List, Dict
import random

class UnderwaterEnvironment:
    """Simulates underwater acoustic channel characteristics."""
    
    def __init__(self, width: float = 1000.0, height: float = 1000.0, 
                 depth: float = 200.0, temperature: float = 10.0):
        self.width = width
        self.height = height
        self.depth = depth
        self.temperature = temperature  # Celsius
        self.salinity = 35.0  # PSU (Practical Salinity Units)
        self.current_time = 0.0
        
    def calculate_sound_speed(self, depth: float) -> float:
        """Calculate sound speed using Mackenzie equation."""
        T = self.temperature
        S = self.salinity
        D = depth
        
        c = (1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3 +
             1.340*(S-35) + 1.630e-2*D + 1.675e-7*D**2 -
             1.025e-2*T*(S-35) - 7.139e-13*T*D**3)
        
        return c
    
    def calculate_path_loss(self, distance: float, frequency: float = 25.0) -> float:
        """Calculate path loss using Thorp's formula."""
        if distance <= 0:
            return 0.0
        
        # Spreading loss (spherical)
        spreading_loss = 20 * np.log10(distance)
        
        # Absorption coefficient (Thorp's approximation)
        f_khz = frequency  # kHz
        absorption = (0.11 * f_khz**2) / (1 + f_khz**2) + \
                    (44 * f_khz**2) / (4100 + f_khz**2) + \
                    2.75e-4 * f_khz**2 + 0.003
        
        # Absorption loss
        absorption_loss = absorption * (distance / 1000.0)  # dB/km
        
        total_loss = spreading_loss + absorption_loss
        return total_loss
    
    def calculate_propagation_delay(self, distance: float, depth: float) -> float:
        """Calculate signal propagation delay."""
        sound_speed = self.calculate_sound_speed(depth)
        delay = distance / sound_speed
        return delay
    
    def add_noise(self, signal_power: float) -> float:
        """Add underwater ambient noise."""
        # Wenz curves approximation for different noise sources
        # Turbulence noise
        turbulence = 17 - 30 * np.log10(25)  # 25 kHz reference
        
        # Shipping noise
        shipping = 40 + 20 * (self.salinity / 35) - 18 * np.log10(25)
        
        # Wind noise  
        wind_speed = random.uniform(0, 10)  # m/s
        wind = 50 + 7.5 * np.sqrt(wind_speed) + 20 * np.log10(25) - 40 * np.log10(25)
        
        # Thermal noise
        thermal = -15 + 20 * np.log10(25)
        
        # Total noise (combine all sources)
        total_noise_db = 10 * np.log10(
            10**(turbulence/10) + 10**(shipping/10) + 
            10**(wind/10) + 10**(thermal/10)
        )
        
        noise_power = 10**(total_noise_db / 10)
        noisy_signal = signal_power + noise_power
        
        return noisy_signal
    
    def calculate_multipath_effect(self, distance: float) -> float:
        """Simulate multipath fading effect."""
        # Rayleigh fading approximation
        fading = np.random.rayleigh(scale=1.0)
        return fading
    
    def get_channel_quality(self, distance: float, frequency: float = 25.0) -> Dict:
        """Calculate overall channel quality metrics."""
        path_loss = self.calculate_path_loss(distance, frequency)
        delay = self.calculate_propagation_delay(distance, self.depth / 2)
        
        # Signal to Noise Ratio (SNR) estimation
        signal_power = 100 - path_loss  # Assuming 100 dB transmission power
        noisy_signal = self.add_noise(10**(signal_power/10))
        snr = 10 * np.log10(10**(signal_power/10) / (noisy_signal - 10**(signal_power/10)))
        
        # Bit Error Rate (BER) approximation for BPSK
        ber = 0.5 * np.exp(-snr / 10) if snr > 0 else 0.5
        
        return {
            'path_loss': path_loss,
            'delay': delay,
            'snr': max(snr, 0),
            'ber': ber,
            'sound_speed': self.calculate_sound_speed(self.depth / 2)
        }
    
    def is_in_range(self, pos1: Tuple[float, float, float], 
                   pos2: Tuple[float, float, float], 
                   max_range: float) -> bool:
        """Check if two nodes are within communication range."""
        distance = self.calculate_distance(pos1, pos2)
        return distance <= max_range
    
    @staticmethod
    def calculate_distance(pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D positions."""
        return np.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
    
    def update_environment(self, time_step: float):
        """Update environmental parameters over time."""
        self.current_time += time_step
        
        # Simulate temperature variations
        temp_variation = 0.1 * np.sin(2 * np.pi * self.current_time / 86400)  # Daily cycle
        self.temperature += temp_variation
        
    def get_random_position(self) -> Tuple[float, float, float]:
        """Generate random 3D position within environment bounds."""
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        z = random.uniform(0, self.depth)
        return (x, y, z)
