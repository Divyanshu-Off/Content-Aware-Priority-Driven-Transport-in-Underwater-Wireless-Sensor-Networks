"""Data Validation and Verification Utilities."""
from typing import Any, Dict, List, Optional

class DataValidator:
    """Validates data integrity and format."""
    
    @staticmethod
    def validate_packet(packet: Dict) -> bool:
        """Validate packet structure and data."""
        required_fields = ['id', 'priority', 'size']
        
        for field in required_fields:
            if field not in packet:
                return False
        
        if not isinstance(packet['priority'], int) or not (0 <= packet['priority'] <= 3):
            return False
        
        if not isinstance(packet['size'], int) or packet['size'] <= 0:
            return False
        
        return True
    
    @staticmethod
    def validate_node_config(config: Dict) -> bool:
        """Validate node configuration."""
        required = ['node_id', 'energy', 'x', 'y']
        
        for field in required:
            if field not in config:
                return False
        
        if config['energy'] < 0 or config['energy'] > 100:
            return False
        
        return True
    
    @staticmethod
    def sanitize_input(data: Any) -> Any:
        """Sanitize input data for safety."""
        if isinstance(data, str):
            return data.strip()[:1000]
        elif isinstance(data, (int, float)):
            return max(0, min(data, 1e6))
        return data
    
    @staticmethod
    def verify_route_validity(route: List[int], topology: Dict) -> bool:
        """Verify route exists in topology."""
        for i in range(len(route) - 1):
            if route[i+1] not in topology.get(route[i], []):
                return False
        return True
