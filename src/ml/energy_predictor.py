"""Energy Consumption Prediction Model for UWSN."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class EnergyPredictor:
    """Predicts energy consumption using machine learning."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'packet_size', 'distance', 'priority', 
            'hop_count', 'link_quality', 'buffer_occupancy'
        ]
    
    def prepare_features(self, data: Dict) -> np.ndarray:
        """Extract and prepare features from data."""
        features = []
        for name in self.feature_names:
            features.append(data.get(name, 0))
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the energy prediction model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        score = self.model.score(X_scaled, y)
        return {'r2_score': score, 'n_samples': len(X)}
    
    def predict(self, data: Dict) -> float:
        """Predict energy consumption for given parameters."""
        if not self.is_trained:
            return self._default_prediction(data)
        
        features = self.prepare_features(data)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        return max(0, prediction)
    
    def _default_prediction(self, data: Dict) -> float:
        """Fallback prediction when model not trained."""
        base_cost = 0.5
        size_factor = data.get('packet_size', 100) / 100.0
        distance_factor = np.log(data.get('distance', 10) + 1)
        return base_cost * size_factor * distance_factor
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
