"""Model Training and Evaluation Utilities."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pickle

class ModelTrainer:
    """Train and evaluate ML models for UWSN."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.training_history = []
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train a model and return metrics."""
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        metrics = {
            'train_score': train_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(X_train)
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                       task: str = 'regression') -> Dict:
        """Evaluate model on test set."""
        y_pred = model.predict(X_test)
        
        if task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            return {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'n_test': len(X_test)
            }
        else:
            accuracy = accuracy_score(y_test, y_pred)
            return {
                'accuracy': accuracy,
                'n_test': len(X_test)
            }
    
    def save_model(self, model, filepath: str) -> bool:
        """Save trained model to disk."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def get_training_summary(self) -> Dict:
        """Get summary of all training sessions."""
        if not self.training_history:
            return {}
        
        return {
            'total_sessions': len(self.training_history),
            'best_cv_score': max(h['cv_mean'] for h in self.training_history),
            'avg_train_score': np.mean([h['train_score'] for h in self.training_history])
        }
