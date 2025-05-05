import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )

    def remove_anomalies(self, X, y):
        """Remove anomalies from the dataset."""
        logger.info("Detecting and removing anomalies...")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Detect anomalies
        anomaly_scores = self.anomaly_detector.fit_predict(X)
        normal_mask = anomaly_scores == 1
        
        # Remove anomalies
        X_clean = X[normal_mask]
        y_clean = y[normal_mask]
        
        n_removed = len(X) - len(X_clean)
        logger.info(f"Removed {n_removed} anomalies ({n_removed/len(X)*100:.2f}% of data)")
        
        return X_clean, y_clean

    def scale_features(self, X):
        """Scale features using StandardScaler."""
        logger.info("Scaling features...")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle zero standard deviation
        X_scaled = self.scaler.fit_transform(X)
        
        # Replace any NaN values with 0
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        return X_scaled

    def encode_labels(self, y):
        """Encode labels using LabelEncoder."""
        logger.info("Encoding labels...")
        
        # Convert to numpy array if needed
        if isinstance(y, pd.Series):
            y = y.values
        
        y_encoded = self.label_encoder.fit_transform(y)
        return y_encoded

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Prepare data for training and testing."""
        logger.info("Preparing data...")
        
        # Remove anomalies
        X_clean, y_clean = self.remove_anomalies(X, y)
        
        # Scale features
        X_scaled = self.scale_features(X_clean)
        
        # Encode labels
        y_encoded = self.encode_labels(y_clean)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, self

def prepare_data(X, y, test_size=0.2, random_state=42):
    """Convenience function to prepare data."""
    preprocessor = DataPreprocessor()
    return preprocessor.prepare_data(X, y, test_size, random_state)

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(X, y)
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}") 