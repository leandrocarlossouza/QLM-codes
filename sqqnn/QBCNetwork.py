"""
Binary Classification using SQQNN
This class implements a binary classifier using the quantum-inspired QNetwork as its base.
It provides both regression and classification capabilities with threshold-based labeling.
"""

from QNetwork import *  # Import base quantum network class
import numpy as np

class QBCNetwork():
    """Quantum Binary Classifier Network that wraps a QNetwork for classification tasks."""
    
    def __init__(self, xt, yt, xv, yv, max_neurons_number):
        """
        Initialize the binary classifier network.
        
        Args:
            xt: Training features (n_samples x n_features)
            yt: Training labels
            xv: Validation features
            yv: Validation labels
            max_neurons_number: Maximum number of neurons/polynomial degree
        """
        # Initialize base quantum network
        self.net = QNetwork(xt, yt, xv, yv, max_neurons_number)
        
        # Store label statistics for classification
        self.ymin = np.min(yt)  # Minimum class label (e.g., -1 or 0)
        self.ymax = np.max(yt)  # Maximum class label (e.g., 1)
        self.cutoff = (self.ymin + self.ymax)/2  # Decision boundary threshold

    def predict(self, x):
        """
        Make continuous predictions (regression output).
        
        Args:
            x: Input features to predict on
            
        Returns:
            Continuous predictions from the quantum network
        """
        return self.net.predict(x)

    def predict_labels(self, x):
        """
        Make discrete class predictions using the cutoff threshold.
        
        Args:
            x: Input features to classify
            
        Returns:
            Array of class predictions (ymin or ymax)
        """
        yh = self.predict(x)  # Get continuous predictions
        # Apply threshold to get class labels
        return np.array([self.ymin if v < self.cutoff else self.ymax for v in yh]).T

    def fit_regression(self, scale=1e-3):
        """
        Fit the model using regression approach with pseudo-inverse.
        
        Args:
            scale: Small scaling factor for numerical stability
            
        Returns:
            Error from the base network's regression fit
        """
        # Preprocess data and compute pseudo-inverse
        nxt = self.net.preprocess_data(self.net.xt)
        Xt = np.transpose(nxt)
        pH = np.matmul(np.linalg.pinv(np.matmul(Xt, nxt), hermitian=True), Xt)
        
        # Fit using the base network's regression method
        return self.net.fit_regression(scale)
        
    def get_errors(self):
        """
        Get training and validation error histories.
        
        Returns:
            Tuple of (training_errors, validation_errors) from the base network
        """
        return self.net.training_errors, self.net.validation_errors