"""Deep learning model for traffic prediction in UWSN."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List

class TrafficPredictor:
    """LSTM-based traffic prediction for underwater sensor networks."""
    
    def __init__(self, sequence_length: int = 50, num_features: int = 6):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self._build_model()
        
    def _build_model(self) -> keras.Model:
        """Build LSTM model for traffic prediction."""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(self.sequence_length, self.num_features)),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32) -> keras.callbacks.History:
        """Train the traffic prediction model."""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict future traffic patterns."""
        return self.model.predict(X)
    
    def predict_next_n_steps(self, initial_sequence: np.ndarray, 
                            n_steps: int) -> List[float]:
        """Predict multiple future time steps."""
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(n_steps):
            # Predict next step
            next_pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, self.num_features))[0][0]
            predictions.append(next_pred)
            
            # Update sequence (rolling window)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = next_pred
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance."""
        loss, mae, mape = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': loss,
            'mae': mae,
            'mape': mape
        }
    
    def save_model(self, filepath: str):
        """Save trained model."""
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load trained model."""
        self.model = keras.models.load_model(filepath)


class CNNLSTMTrafficPredictor:
    """CNN-LSTM hybrid model for enhanced traffic pattern recognition."""
    
    def __init__(self, sequence_length: int = 50, num_features: int = 6):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build CNN-LSTM hybrid model."""
        model = keras.Sequential([
            layers.Conv1D(64, kernel_size=3, activation='relu',
                         input_shape=(self.sequence_length, self.num_features)),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(32, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32) -> keras.callbacks.History:
        """Train the CNN-LSTM model."""
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
