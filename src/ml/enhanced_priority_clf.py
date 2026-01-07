"""Enhanced Priority Classifier using XGBoost for UWSN Packet Classification

This module implements an improved machine learning model for classifying
packets into priority levels in Underwater Wireless Sensor Networks (UWSNs).
It uses XGBoost with class weighting to handle imbalanced priority distributions.

Author: UWSN Project Team
Date: 2026
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import json
from typing import Dict, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPriorityClassifier:
    """XGBoost-based priority classifier for UWSN packets."""

    def __init__(self, random_state: int = 42):
        """Initialize the classifier.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.class_weights = None
        self.model_params = {}
        self.training_history = {}

    def calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate weights for imbalanced classes.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary of class weights
        """
        unique, counts = np.unique(y, return_counts=True)
        weights = {}
        total = len(y)
        
        for label, count in zip(unique, counts):
            # Weight is inversely proportional to class frequency
            weights[label] = total / (len(unique) * count)
        
        logger.info(f"Calculated class weights: {weights}")
        return weights

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              val_split: float = 0.2, verbose: bool = True) -> Dict:
        """Train the XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            val_split: Validation split ratio
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        # Store feature names
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Calculate class weights
        self.class_weights = self.calculate_class_weights(y_encoded)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Get number of classes
        n_classes = len(np.unique(y_encoded))
        
        # XGBoost parameters optimized for UWSN
        self.model_params = {
            'objective': 'multi:softmax' if n_classes > 2 else 'binary:logistic',
            'num_class': n_classes if n_classes > 2 else 2,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 200,
            'random_state': self.random_state,
            'scale_pos_weight': self.class_weights.get(1, 1),
            'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
        }
        
        # Create and train classifier
        self.classifier = xgb.XGBClassifier(**self.model_params)
        
        # Train with validation set
        eval_set = [(X_scaled, y_encoded)]
        
        self.classifier.fit(
            X_scaled, y_encoded,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Store training history
        self.training_history = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_classes': n_classes,
            'feature_names': self.feature_names,
            'class_distribution': dict(zip(*np.unique(y_encoded, return_counts=True))),
            'model_params': self.model_params
        }
        
        logger.info(f"Training completed. Training history: {self.training_history}")
        return self.training_history

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """Evaluate the classifier on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Encode labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        y_pred = self.classifier.predict(X_test_scaled)
        y_pred_proba = self.classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y_test_encoded, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test_encoded, y_pred).tolist(),
            'f1_score_weighted': f1_score(y_test_encoded, y_pred, average='weighted'),
            'f1_score_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        }
        
        # Try to calculate ROC-AUC for binary or one-vs-rest
        try:
            if len(np.unique(y_test_encoded)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
            else:
                metrics['roc_auc_ovr'] = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def predict(self, X: pd.DataFrame, return_proba: bool = False):
        """Make predictions on new data.
        
        Args:
            X: Input features
            return_proba: Whether to return prediction probabilities
            
        Returns:
            Predictions (and probabilities if requested)
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.classifier.predict(X_scaled)
        
        if return_proba:
            proba = self.classifier.predict_proba(X_scaled)
            return predictions, proba
        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importance = self.classifier.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'relative_importance': importance / importance.sum()
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def save_model(self, filepath: str):
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.classifier is None:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        self.model_params = model_data['model_params']
        logger.info(f"Model loaded from {filepath}")

    def cross_validate(self, X: pd.DataFrame, y: np.ndarray,
                       cv_folds: int = 5) -> Dict:
        """Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Setup cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Define scoring metrics
        scoring = {
            'f1_weighted': 'f1_weighted',
            'f1_macro': 'f1_macro',
            'accuracy': 'accuracy'
        }
        
        # Create classifier
        clf = xgb.XGBClassifier(**self.model_params)
        
        # Perform cross-validation
        cv_results = cross_validate(clf, X_scaled, y_encoded, cv=skf, scoring=scoring)
        
        # Calculate mean and std
        cv_summary = {}
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            cv_summary[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        
        logger.info(f"Cross-validation results: {cv_summary}")
        return cv_summary
