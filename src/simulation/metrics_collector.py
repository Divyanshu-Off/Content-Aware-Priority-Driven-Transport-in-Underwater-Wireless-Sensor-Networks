"""Metrics collection for simulation analysis."""

import numpy as np
from typing import Dict, List
import json

class MetricsCollector:
    """Collects and analyzes simulation metrics."""
    
    def __init__(self):
        self.metrics = {
            'packets_sent': [],
            'packets_received': [],
            'packets_dropped': [],
            'end_to_end_delay': [],
            'energy_consumed': [],
            'throughput': [],
            'delivery_ratio': [],
            'network_lifetime': 0.0
        }
        self.time_series = []
        
    def record_packet_sent(self, time: float, node_id: int, packet_id: int):
        """Record packet transmission."""
        self.metrics['packets_sent'].append({
            'time': time,
            'node_id': node_id,
            'packet_id': packet_id
        })
        
    def record_packet_received(self, time: float, node_id: int, 
                              packet_id: int, delay: float):
        """Record packet reception."""
        self.metrics['packets_received'].append({
            'time': time,
            'node_id': node_id,
            'packet_id': packet_id,
            'delay': delay
        })
        self.metrics['end_to_end_delay'].append(delay)
        
    def record_packet_dropped(self, time: float, node_id: int, 
                             packet_id: int, reason: str):
        """Record packet drop."""
        self.metrics['packets_dropped'].append({
            'time': time,
            'node_id': node_id,
            'packet_id': packet_id,
            'reason': reason
        })
        
    def record_energy_consumption(self, time: float, node_id: int, 
                                 energy: float):
        """Record energy consumption."""
        self.metrics['energy_consumed'].append({
            'time': time,
            'node_id': node_id,
            'energy': energy
        })
        
    def record_snapshot(self, time: float, network_stats: Dict):
        """Record network snapshot at specific time."""
        snapshot = {'time': time}
        snapshot.update(network_stats)
        self.time_series.append(snapshot)
        
    def calculate_statistics(self) -> Dict:
        """Calculate overall statistics."""
        total_sent = len(self.metrics['packets_sent'])
        total_received = len(self.metrics['packets_received'])
        total_dropped = len(self.metrics['packets_dropped'])
        
        # Packet Delivery Ratio
        pdr = (total_received / total_sent * 100) if total_sent > 0 else 0
        
        # Average End-to-End Delay
        avg_delay = (np.mean(self.metrics['end_to_end_delay']) 
                    if self.metrics['end_to_end_delay'] else 0)
        
        # Average Energy Consumption
        energy_values = [e['energy'] for e in self.metrics['energy_consumed']]
        avg_energy = np.mean(energy_values) if energy_values else 0
        total_energy = np.sum(energy_values) if energy_values else 0
        
        # Throughput (packets per second)
        if self.time_series:
            duration = self.time_series[-1]['time'] - self.time_series[0]['time']
            throughput = total_received / duration if duration > 0 else 0
        else:
            throughput = 0
            
        return {
            'total_packets_sent': total_sent,
            'total_packets_received': total_received,
            'total_packets_dropped': total_dropped,
            'packet_delivery_ratio': pdr,
            'avg_end_to_end_delay': avg_delay,
            'avg_energy_consumption': avg_energy,
            'total_energy_consumed': total_energy,
            'throughput': throughput,
            'network_lifetime': self.metrics['network_lifetime']
        }
        
    def get_time_series_data(self) -> List[Dict]:
        """Get time series data for plotting."""
        return self.time_series
    
    def export_to_json(self, filepath: str):
        """Export metrics to JSON file."""
        stats = self.calculate_statistics()
        export_data = {
            'statistics': stats,
            'time_series': self.time_series,
            'raw_metrics': {
                'packets_sent': len(self.metrics['packets_sent']),
                'packets_received': len(self.metrics['packets_received']),
                'packets_dropped': len(self.metrics['packets_dropped'])
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=4)
            
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'packets_sent': [],
            'packets_received': [],
            'packets_dropped': [],
            'end_to_end_delay': [],
            'energy_consumed': [],
            'throughput': [],
            'delivery_ratio': [],
            'network_lifetime': 0.0
        }
        self.time_series = []
