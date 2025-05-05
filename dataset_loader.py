import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import os
from sklearn.datasets import fetch_covtype

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkDatasetLoader:
    def __init__(self):
        self.dataset_name = "Forest Cover Type"
        self.label_encoder = LabelEncoder()

    def load_and_prepare_data(self, sample_size=None):
        """Load and prepare the Forest Cover Type dataset."""
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        try:
            # Load the dataset
            data = fetch_covtype()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = data.target
            
            # Convert labels to start from 0
            y = y - 1
            
            # Handle missing values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            if sample_size:
                # Take a random sample if specified
                indices = np.random.choice(len(X), sample_size, replace=False)
                X = X.iloc[indices]
                y = y[indices]
            
            logger.info(f"Dataset loaded successfully. Shape: {X.shape}")
            logger.info(f"Number of classes: {len(np.unique(y))}")
            logger.info("Class distribution:")
            for label in np.unique(y):
                count = np.sum(y == label)
                percentage = (count / len(y)) * 100
                logger.info(f"Class {label}: {count} samples ({percentage:.2f}%)")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None, None

def get_network_dataset(sample_size=None):
    """Helper function to get the network dataset."""
    loader = NetworkDatasetLoader()
    return loader.load_and_prepare_data(sample_size)

if __name__ == "__main__":
    # Example usage
    X, y = get_network_dataset(sample_size=10000)  # Load 10,000 samples for testing
    if X is not None and y is not None:
        logger.info(f"Successfully loaded dataset with {X.shape[0]} samples and {X.shape[1]} features") 