import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
import tensorflow as tf
import logging
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityTester:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.art_classifier = model.art_classifier

    def test_evasion_attack(self, X_test, y_test, attack_type='fgsm'):
        """Test the model against evasion attacks."""
        logger.info(f"Testing {attack_type} evasion attack...")
        
        if attack_type == 'fgsm':
            attack = FastGradientMethod(
                estimator=self.art_classifier,
                eps=0.1,
                batch_size=64
            )
        elif attack_type == 'pgd':
            attack = ProjectedGradientDescent(
                estimator=self.art_classifier,
                eps=0.1,
                eps_step=0.01,
                max_iter=10,
                batch_size=64
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Generate adversarial examples
        X_test_adv = attack.generate(x=X_test)
        
        # Evaluate original accuracy
        y_pred_orig = self.model.predict(X_test)
        acc_orig = accuracy_score(y_test, y_pred_orig)
        
        # Evaluate accuracy on adversarial examples
        y_pred_adv = self.model.predict(X_test_adv)
        acc_adv = accuracy_score(y_test, y_pred_adv)
        
        logger.info(f"Original accuracy: {acc_orig:.4f}")
        logger.info(f"Accuracy under attack: {acc_adv:.4f}")
        
        return acc_orig, acc_adv, X_test_adv

    def test_poisoning_attack(self, X_train, y_train, X_test, y_test, poison_percentage=0.1):
        """Test the model against poisoning attacks."""
        logger.info("Testing poisoning attack...")
        
        # Create poisoned data by adding noise to a subset of training data
        n_samples = len(X_train)
        n_poison = int(n_samples * poison_percentage)
        
        # Randomly select samples to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        # Create poisoned data by adding noise
        X_train_poisoned = X_train.copy()
        noise = np.random.normal(0, 0.1, X_train[poison_indices].shape)
        X_train_poisoned[poison_indices] += noise
        
        # Ensure values stay in valid range
        X_train_poisoned = np.clip(X_train_poisoned, 0, 1)
        
        # Train model on poisoned data
        self.model.train_ensemble(X_train_poisoned, y_train)
        self.model.train_neural_network(X_train_poisoned, y_train)
        
        # Evaluate on clean test data
        y_pred = self.model.predict(X_test)
        acc_poisoned = accuracy_score(y_test, y_pred)
        
        logger.info(f"Accuracy after poisoning attack: {acc_poisoned:.4f}")
        
        return acc_poisoned

    def visualize_attack_results(self, X_test, X_test_adv, attack_type):
        """Visualize the effects of attacks on the data."""
        plt.figure(figsize=(12, 5))
        
        # Use PCA to reduce dimensions for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_test_2d = pca.fit_transform(X_test)
        X_test_adv_2d = pca.transform(X_test_adv)
        
        # Plot original vs adversarial examples
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=X_test_2d[:, 0], y=X_test_2d[:, 1], alpha=0.5, label='Original')
        plt.title('Original Data')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=X_test_adv_2d[:, 0], y=X_test_adv_2d[:, 1], alpha=0.5, label='Adversarial')
        plt.title(f'{attack_type.upper()} Attack')
        
        plt.tight_layout()
        plt.savefig(f'{attack_type}_attack_visualization.png')
        plt.close()

def run_security_tests(model, preprocessor, X_train, y_train, X_test, y_test):
    """Run comprehensive security tests."""
    tester = SecurityTester(model, preprocessor)
    
    # Test evasion attacks
    fgsm_results = tester.test_evasion_attack(X_test, y_test, 'fgsm')
    pgd_results = tester.test_evasion_attack(X_test, y_test, 'pgd')
    
    # Test poisoning attack
    poisoning_results = tester.test_poisoning_attack(X_train, y_train, X_test, y_test)
    
    # Visualize results
    tester.visualize_attack_results(X_test, fgsm_results[2], 'fgsm')
    
    return {
        'fgsm_attack': fgsm_results[:2],
        'pgd_attack': pgd_results[:2],
        'poisoning_attack': poisoning_results
    }

if __name__ == "__main__":
    # Import the dataset loader
    from dataset_loader import get_network_dataset
    from data_preprocessing import prepare_data
    from model_training import train_robust_model
    
    # Load the network dataset
    logger.info("Loading network dataset...")
    X, y = get_network_dataset(sample_size=50000)
    
    if X is None or y is None:
        logger.error("Failed to load dataset. Please make sure you have downloaded the dataset.")
        exit(1)
    
    # Prepare data
    logger.info("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(X, y)
    
    # Train model
    logger.info("Training robust model...")
    model, accuracy, report = train_robust_model(X_train, y_train, X_test, y_test)
    
    # Run security tests
    logger.info("Running security tests...")
    security_results = run_security_tests(model, preprocessor, X_train, y_train, X_test, y_test)
    
    # Print results
    logger.info("\nSecurity Test Results:")
    for attack, results in security_results.items():
        if isinstance(results, tuple):
            logger.info(f"{attack}: Original accuracy = {results[0]:.4f}, "
                       f"Attack accuracy = {results[1]:.4f}")
        else:
            logger.info(f"{attack}: Accuracy = {results:.4f}") 