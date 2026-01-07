"""Synthetic UWSN Traffic Generator

Generates realistic packet traffic matching Phase 1 specification.
Phase 1 Reference: https://github.com/Divyanshu-Off/Content-Aware-Priority-Driven-Transport-in-Underwater-Wireless-Sensor-Networks/blob/main/docs/PHASE_1_DESIGN.md
"""

import random
import math
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PacketRecord:
    """Single packet record matching Phase 1 packet schema (§3)."""
    packet_id: int
    timestamp: float
    src_node_id: int
    dst_node_id: int
    class_id: int  # 0=C, 1=B, 2=A
    payload_size: int
    is_anomaly: bool
    depth: float
    residual_energy: float
    sensed_value: float = 20.0  # default baseline


class TrafficGenerator:
    """Generates synthetic UWSN traffic per Phase 1 spec.
    
    Traffic Classes (Phase 1 §2.1):
    - Class A (emergency): λ=0.1 pkt/node/sec, 128 bytes, urgent
    - Class B (control): λ=0.01 pkt/node/sec, 64 bytes, moderate
    - Class C (routine): λ=0.02 pkt/node/sec, 256 bytes, flexible
    """
    
    # Phase 1 Parameters
    LAMBDA_A = 0.1    # Poisson rate for Class A (packets/node/sec)
    LAMBDA_B = 0.01   # Poisson rate for Class B
    LAMBDA_C = 0.02   # Poisson rate for Class C
    
    PAYLOAD_SIZES = {0: 256, 1: 64, 2: 128}  # bytes per class (C, B, A)
    MAX_NODES = 50
    SINK_NODE = 50
    
    # Network parameters
    MIN_DEPTH = 0.0
    MAX_DEPTH = 500.0
    MIN_ENERGY = 5000.0
    MAX_ENERGY = 10000.0
    
    # Anomaly detection: if sensed > mean + 2*sigma
    SENSOR_MEAN = 20.0
    SENSOR_STD = 2.0
    ANOMALY_THRESHOLD = SENSOR_MEAN + 2 * SENSOR_STD
    
    def __init__(self, seed: int = 42):
        """Initialize generator with optional seed for reproducibility."""
        random.seed(seed)
        self.packet_count = 0
    
    def generate_traffic(self, num_nodes: int, num_packets: int, 
                        output_file: str = None) -> List[PacketRecord]:
        """Generate synthetic traffic dataset.
        
        Args:
            num_nodes: Number of sensor nodes (max 50)
            num_packets: Total packets to generate (~5000 recommended)
            output_file: Optional CSV output path
        
        Returns:
            List of PacketRecord objects
        """
        assert num_nodes <= self.MAX_NODES, f"Max nodes is {self.MAX_NODES}"
        
        packets = []
        self.packet_count = 0
        
        # Generate packets with Poisson arrivals
        # Total expected per node over 1000 time units:
        # A: 0.1*1000=100, B: 0.01*1000=10, C: 0.02*1000=20 -> 130/node
        # For 50 nodes: ~6500 total packets
        # So for num_packets=5000, we have ~77 packets/node
        
        pkts_per_node = num_packets // num_nodes
        
        for node_id in range(num_nodes):
            current_time = 0.0
            
            # Generate interarrival times using exponential distribution
            # Combine all three classes
            total_lambda = self.LAMBDA_A + self.LAMBDA_B + self.LAMBDA_C
            
            while len(packets) < num_packets and current_time < 1000.0:
                # Exponential interarrival time
                inter_arrival = random.expovariate(total_lambda)
                current_time += inter_arrival
                
                if current_time < 1000.0 and len(packets) < num_packets:
                    # Determine class based on weighted random selection
                    class_id = self._select_class()
                    
                    # Create packet
                    pkt = self._create_packet(
                        node_id=node_id,
                        timestamp=current_time,
                        class_id=class_id
                    )
                    packets.append(pkt)
        
        # Sort by timestamp
        packets.sort(key=lambda p: p.timestamp)
        packets = packets[:num_packets]  # Trim to exact count
        
        # Verify class distribution
        dist = self._compute_distribution(packets)
        print(f"\n[TrafficGenerator] Generated {len(packets)} packets")
        print(f"  Class A (emergency): {dist[2]} ({100*dist[2]/len(packets):.1f}%)")
        print(f"  Class B (control): {dist[1]} ({100*dist[1]/len(packets):.1f}%)")
        print(f"  Class C (routine): {dist[0]} ({100*dist[0]/len(packets):.1f}%)")
        print(f"  Anomalies: {sum(1 for p in packets if p.is_anomaly)}")
        
        # Save to CSV if requested
        if output_file:
            self._save_to_csv(packets, output_file)
        
        return packets
    
    def _select_class(self) -> int:
        """Randomly select packet class based on Poisson rates."""
        r = random.random()
        total_lambda = self.LAMBDA_A + self.LAMBDA_B + self.LAMBDA_C
        
        # Normalized probabilities
        p_a = self.LAMBDA_A / total_lambda
        p_b = self.LAMBDA_B / total_lambda
        
        if r < p_a:
            return 2  # Class A
        elif r < p_a + p_b:
            return 1  # Class B
        else:
            return 0  # Class C
    
    def _create_packet(self, node_id: int, timestamp: float, 
                      class_id: int) -> PacketRecord:
        """Create a single packet with realistic parameters."""
        self.packet_count += 1
        
        # Random sensor reading (normal distribution)
        sensed_value = random.gauss(self.SENSOR_MEAN, self.SENSOR_STD)
        is_anomaly = sensed_value > self.ANOMALY_THRESHOLD
        
        # Random depth
        depth = random.uniform(self.MIN_DEPTH, self.MAX_DEPTH)
        
        # Random residual energy
        residual_energy = random.uniform(self.MIN_ENERGY, self.MAX_ENERGY)
        
        pkt = PacketRecord(
            packet_id=self.packet_count,
            timestamp=timestamp,
            src_node_id=node_id,
            dst_node_id=self.SINK_NODE,
            class_id=class_id,
            payload_size=self.PAYLOAD_SIZES[class_id],
            is_anomaly=int(is_anomaly),
            depth=round(depth, 1),
            residual_energy=round(residual_energy, 1),
            sensed_value=round(sensed_value, 2)
        )
        return pkt
    
    @staticmethod
    def _compute_distribution(packets: List[PacketRecord]) -> dict:
        """Count packets by class."""
        dist = {0: 0, 1: 0, 2: 0}
        for pkt in packets:
            dist[pkt.class_id] += 1
        return dist
    
    @staticmethod
    def _save_to_csv(packets: List[PacketRecord], output_file: str):
        """Save packet list to CSV file."""
        data = [{
            'packet_id': p.packet_id,
            'timestamp': p.timestamp,
            'src_node_id': p.src_node_id,
            'dst_node_id': p.dst_node_id,
            'class_id': p.class_id,
            'payload_size': p.payload_size,
            'is_anomaly': p.is_anomaly,
            'depth': p.depth,
            'residual_energy': p.residual_energy,
            'sensed_value': p.sensed_value
        } for p in packets]
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")


if __name__ == '__main__':
    # Example usage
    gen = TrafficGenerator(seed=42)
    packets = gen.generate_traffic(
        num_nodes=50,
        num_packets=5000,
        output_file='syn_data/dataset.csv'
    )
