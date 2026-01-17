"""Feature Extraction Utilities for ML Models."""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class FeatureExtractor:
    """Extract and transform features from network data."""
    
    def __init__(self):
        self.feature_stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'count': 0})
    
    def extract_packet_features(self, packet: Dict) -> Dict[str, float]:
        """Extract features from packet data."""
        features = {
            'size': packet.get('size', 0),
            'priority': packet.get('priority', 1),
            'hop_count': packet.get('hop_count', 0),
            'ttl': packet.get('ttl', 64),
            'timestamp': packet.get('timestamp', 0)
        }
        return features
    
    def extract_node_features(self, node: Dict) -> Dict[str, float]:
        """Extract features from node state."""
        features = {
            'energy_level': node.get('energy', 100),
            'buffer_occupancy': node.get('buffer_occupancy', 0),
            'tx_count': node.get('tx_count', 0),
            'rx_count': node.get('rx_count', 0),
            'neighbor_count': len(node.get('neighbors', []))
        }
        return features
    
    def extract_link_features(self, link_data: Dict) -> Dict[str, float]:
        """Extract features from link quality data."""
        features = {
            'rssi': link_data.get('rssi', -70),
            'snr': link_data.get('snr', 10),
            'packet_loss': link_data.get('packet_loss', 0),
            'delay': link_data.get('delay', 0),
            'distance': link_data.get('distance', 0)
        }
        return features
    
    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features using min-max scaling."""
        normalized = {}
        for key, value in features.items():
            stats = self.feature_stats[key]
            if stats['max'] > stats['min']:
                normalized[key] = (value - stats['min']) / (stats['max'] - stats['min'])
            else:
                normalized[key] = 0.5
        return normalized
    
    def update_statistics(self, features: Dict[str, float]) -> None:
        """Update feature statistics for normalization."""
        for key, value in features.items():
            stats = self.feature_stats[key]
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['count'] += 1
            stats['mean'] = (stats['mean'] * (stats['count'] - 1) + value) / stats['count']
    
    def create_feature_vector(self, data: Dict) -> np.ndarray:
        """Create feature vector from raw data."""
        features = []
        features.extend(self.extract_packet_features(data.get('packet', {})).values())
        features.extend(self.extract_node_features(data.get('node', {})).values())
        features.extend(self.extract_link_features(data.get('link', {})).values())
        return np.array(features)
