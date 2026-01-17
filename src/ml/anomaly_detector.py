"""Anomaly Detection for UWSN Network Behavior."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from collections import deque

class AnomalyDetector:
    """Detects anomalous behavior in network using ML."""
    
    def __init__(self, contamination: float = 0.1, window_size: int = 50):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.is_trained = False
        self.anomaly_count = 0
        self.feature_names = [
            'packet_loss_rate', 'delay', 'energy_consumption',
            'hop_count', 'retransmission_rate', 'buffer_occupancy'
        ]
    
    def extract_features(self, metrics: Dict) -> np.ndarray:
        """Extract features from network metrics."""
        features = []
        for name in self.feature_names:
            features.append(metrics.get(name, 0))
        return np.array(features)
    
    def train(self, historical_data: List[Dict]) -> Dict:
        """Train anomaly detection model on historical data."""
        if len(historical_data) < 10:
            return {'status': 'insufficient_data', 'samples': len(historical_data)}
        
        X = np.array([self.extract_features(d) for d in historical_data])
        self.model.fit(X)
        self.is_trained = True
        
        return {'status': 'trained', 'samples': len(X)}
    
    def detect(self, metrics: Dict) -> Tuple[bool, float]:
        """Detect if current metrics indicate anomaly."""
        features = self.extract_features(metrics).reshape(1, -1)
        self.data_buffer.append(features[0])
        
        if not self.is_trained:
            if len(self.data_buffer) >= self.window_size:
                X = np.array(list(self.data_buffer))
                self.model.fit(X)
                self.is_trained = True
            return False, 0.0
        
        prediction = self.model.predict(features)[0]
        anomaly_score = self.model.score_samples(features)[0]
        
        is_anomaly = prediction == -1
        if is_anomaly:
            self.anomaly_count += 1
        
        return is_anomaly, abs(anomaly_score)
    
    def get_anomaly_rate(self) -> float:
        """Calculate anomaly detection rate."""
        if len(self.data_buffer) == 0:
            return 0.0
        return self.anomaly_count / len(self.data_buffer)
    
    def reset(self) -> None:
        """Reset anomaly detector state."""
        self.data_buffer.clear()
        self.anomaly_count = 0
        self.is_trained = False
