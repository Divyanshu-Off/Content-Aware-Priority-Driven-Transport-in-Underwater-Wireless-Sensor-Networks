"""Data preprocessing utilities for ML models in UWSN."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, List

class DataPreprocessor:
    """Preprocessing pipeline for UWSN data."""
    
    def __init__(self, scaler_type: str = 'standard'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        self.is_fitted = False
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data."""
        scaled_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        return scaled_data
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        return self.scaler.inverse_transform(data)
    
    @staticmethod
    def create_sequences(data: np.ndarray, sequence_length: int,
                        step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X, y = [], []
        for i in range(0, len(data) - sequence_length, step):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, 0])  # Predict first feature
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def split_train_val_test(data: np.ndarray, 
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets."""
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame, 
                            method: str = 'interpolate') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if method == 'interpolate':
            return data.interpolate(method='linear')
        elif method == 'forward_fill':
            return data.fillna(method='ffill')
        elif method == 'backward_fill':
            return data.fillna(method='bfill')
        elif method == 'mean':
            return data.fillna(data.mean())
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def remove_outliers(data: np.ndarray, 
                       threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using z-score method."""
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        mask = np.all(z_scores < threshold, axis=1)
        return data[mask]


class FeatureEngineering:
    """Feature engineering for UWSN data."""
    
    @staticmethod
    def extract_statistical_features(data: np.ndarray, 
                                    window_size: int = 10) -> np.ndarray:
        """Extract statistical features from time series data."""
        features = []
        
        for i in range(window_size, len(data)):
            window = data[i - window_size:i]
            
            feature_vector = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.median(window),
                np.percentile(window, 25),
                np.percentile(window, 75)
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    @staticmethod
    def calculate_energy_features(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate energy-related features."""
        df = data.copy()
        
        # Energy consumption rate
        if 'energy' in df.columns:
            df['energy_rate'] = df['energy'].diff().fillna(0)
            df['energy_ma'] = df['energy'].rolling(window=5).mean()
        
        # Energy efficiency
        if 'packets_sent' in df.columns and 'energy' in df.columns:
            df['energy_per_packet'] = df['energy'] / (df['packets_sent'] + 1)
        
        return df
    
    @staticmethod
    def calculate_network_features(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate network-related features."""
        df = data.copy()
        
        # Throughput
        if 'packets_sent' in df.columns:
            df['throughput'] = df['packets_sent'].rolling(window=5).sum()
        
        # Delay variation
        if 'delay' in df.columns:
            df['delay_variance'] = df['delay'].rolling(window=5).var()
            df['delay_ma'] = df['delay'].rolling(window=5).mean()
        
        # Packet loss rate
        if 'packets_sent' in df.columns and 'packets_received' in df.columns:
            df['packet_loss_rate'] = 1 - (df['packets_received'] / (df['packets_sent'] + 1))
        
        return df


class DataAugmentation:
    """Data augmentation for training ML models."""
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to data."""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    @staticmethod
    def time_shift(data: np.ndarray, shift_range: int = 5) -> np.ndarray:
        """Apply random time shift to data."""
        shift = np.random.randint(-shift_range, shift_range)
        return np.roll(data, shift, axis=0)
    
    @staticmethod
    def scale_data(data: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply random scaling to data."""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale_factor
