import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
import logging
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from art.attacks.evasion import FastGradientMethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustEnsembleModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.ensemble = RandomForestClassifier(
            n_estimators=500,  # Increased number of trees
            max_depth=30,      # Increased depth
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',  # Better handling of class imbalance
            n_jobs=-1,
            random_state=42
        )
        self.nn_model = self._build_nn_model()
        self.art_classifier = None
        self._initialize_art_classifier()

    def _build_nn_model(self):
        """Build a robust neural network model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),  # Reduced learning rate
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _initialize_art_classifier(self):
        """Initialize the ART classifier for the neural network."""
        self.art_classifier = TensorFlowV2Classifier(
            model=self.nn_model,
            nb_classes=self.num_classes,
            input_shape=self.input_shape,
            loss_object=tf.keras.losses.SparseCategoricalCrossentropy(),
            clip_values=(0, 1)
        )

    def train_ensemble(self, X_train, y_train):
        """Train the Random Forest ensemble."""
        logger.info("Training Random Forest ensemble...")
        self.ensemble.fit(X_train, y_train)
        return self.ensemble

    def train_neural_network(self, X_train, y_train, validation_split=0.2, epochs=100):
        """Train the neural network with early stopping, class weights, and adversarial training."""
        logger.info("Training neural network...")
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Increased patience
            restore_best_weights=True
        )
        
        # Add learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Adversarial training setup
        attack = FastGradientMethod(
            estimator=self.art_classifier,
            eps=0.1,
            batch_size=64
        )
        
        # Generate adversarial examples
        X_train_adv = attack.generate(x=X_train)
        
        # Combine original and adversarial examples
        X_train_combined = np.vstack((X_train, X_train_adv))
        y_train_combined = np.hstack((y_train, y_train))
        
        history = self.nn_model.fit(
            X_train_combined, y_train_combined,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,  # Smaller batch size for better generalization
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Reinitialize ART classifier with trained model
        self._initialize_art_classifier()
        return history

    def predict(self, X):
        """Make predictions using both models and combine results."""
        rf_pred = self.ensemble.predict_proba(X)
        nn_pred = self.nn_model.predict(X)
        
        # Combine predictions (weighted average based on validation performance)
        combined_pred = (rf_pred + nn_pred) / 2
        return np.argmax(combined_pred, axis=1)

def train_robust_model(X_train, y_train, X_test, y_test):
    """Train and evaluate the robust ensemble model."""
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))
    
    model = RobustEnsembleModel(input_shape, num_classes)
    
    # Train both components
    model.train_ensemble(X_train, y_train)
    model.train_neural_network(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(report)
    
    return model, accuracy, report

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from data_preprocessing import prepare_data
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(X, y)
    
    # Train and evaluate model
    model, accuracy, report = train_robust_model(X_train, y_train, X_test, y_test)