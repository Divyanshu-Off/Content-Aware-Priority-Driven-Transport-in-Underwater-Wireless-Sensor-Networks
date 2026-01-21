"""Model evaluation and metrics utilities for UWSN ML models."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import json

class ModelEvaluator:
    """Comprehensive model evaluation for UWSN ML models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics = {}
        self.predictions = []
        self.actual_values = []
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate regression model performance."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               labels: Optional[List[str]] = None) -> Dict:
        """Evaluate classification model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=labels,
                                      output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        self.metrics.update(metrics)
        return metrics
    
    def calculate_energy_efficiency_metrics(self, energy_consumed: np.ndarray,
                                          packets_delivered: np.ndarray) -> Dict:
        """Calculate energy efficiency metrics for UWSN."""
        total_energy = np.sum(energy_consumed)
        total_packets = np.sum(packets_delivered)
        
        energy_per_packet = total_energy / (total_packets + 1e-8)
        avg_energy_consumption = np.mean(energy_consumed)
        energy_variance = np.var(energy_consumed)
        
        metrics = {
            'total_energy_consumed': float(total_energy),
            'total_packets_delivered': int(total_packets),
            'energy_per_packet': float(energy_per_packet),
            'avg_energy_consumption': float(avg_energy_consumption),
            'energy_variance': float(energy_variance)
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def calculate_network_performance_metrics(self, delays: np.ndarray,
                                            throughputs: np.ndarray,
                                            packet_loss_rates: np.ndarray) -> Dict:
        """Calculate network performance metrics."""
        avg_delay = np.mean(delays)
        max_delay = np.max(delays)
        delay_std = np.std(delays)
        
        avg_throughput = np.mean(throughputs)
        total_throughput = np.sum(throughputs)
        
        avg_packet_loss = np.mean(packet_loss_rates)
        
        metrics = {
            'avg_delay': float(avg_delay),
            'max_delay': float(max_delay),
            'delay_std': float(delay_std),
            'avg_throughput': float(avg_throughput),
            'total_throughput': float(total_throughput),
            'avg_packet_loss_rate': float(avg_packet_loss)
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 save_path: Optional[str] = None):
        """Plot predictions vs actual values."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{self.model_name} - Predictions vs Actual')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None):
        """Plot residuals distribution."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals plot
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals Plot')
        axes[0].grid(True)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=50, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{self.model_name} - Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report."""
        report = f"\n{'='*50}\n"
        report += f"Model Evaluation Report: {self.model_name}\n"
        report += f"{'='*50}\n\n"
        
        for key, value in self.metrics.items():
            if isinstance(value, (int, float)):
                report += f"{key}: {value:.4f}\n"
            elif key != 'confusion_matrix' and key != 'classification_report':
                report += f"{key}: {value}\n"
        
        return report


class CrossValidator:
    """Cross-validation utilities for model evaluation."""
    
    @staticmethod
    def time_series_split(data: np.ndarray, n_splits: int = 5) -> List[Tuple]:
        """Time series cross-validation split."""
        n = len(data)
        test_size = n // (n_splits + 1)
        splits = []
        
        for i in range(n_splits):
            train_end = test_size * (i + 1)
            test_end = train_end + test_size
            
            train_indices = list(range(0, train_end))
            test_indices = list(range(train_end, test_end))
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    @staticmethod
    def calculate_cv_scores(scores: List[float]) -> Dict:
        """Calculate cross-validation statistics."""
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }
