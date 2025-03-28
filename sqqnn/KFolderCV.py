"""
K-Fold Cross Validation for SQQNN
This class implements k-fold cross validation to evaluate the performance of the QBCNetwork model.
"""

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from QBCNetwork import *

class KFolderCV:
    """
    K-Fold Cross Validator for Quantum Binary Classification Network.
    
    This class performs k-fold cross validation and computes multiple evaluation metrics
    including accuracy, precision, sensitivity (recall), specificity, and F1 score.
    """
    
    def __init__(self, n_splits=10):
        """
        Initialize the k-fold cross validator.
        
        Args:
            n_splits (int): Number of folds for cross validation (default=10)
        """
        self.n_splits = n_splits

    def cross_validate(self, X, y, nk):
        """
        Perform k-fold cross validation on the QBCNetwork model.
        
        Args:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
            y (numpy.ndarray): Target labels of shape (n_samples,)
            nk (int): Number of neurons/polynomial degree for QBCNetwork
            
        Returns:
            tuple: Contains means and standard deviations of:
                - accuracy
                - precision  
                - sensitivity (recall)
                - specificity
                - F1 score
        """
        # Initialize k-fold splitter
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # Initialize lists to store metrics for each fold
        accuracy = []
        precision = []
        sensivity = []  # Note: Typically spelled "sensitivity"
        specificity = []
        F1 = []
        
        # Perform k-fold cross validation
        for train_index, test_index in kf.split(X):
            # Split data into training and test sets
            X_train, X_test = X[train_index,:], X[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            
            # Initialize and train quantum binary classifier
            model = QBCNetwork(X_train, y_train, X_train, y_train, nk)
            model.fit_regression(1e-16)  # Fit with very small scale factor
            
            # Make predictions on test set
            y_pred = model.predict_labels(X_test)
            
            # Compute evaluation metrics
            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(metrics.precision_score(y_test, y_pred))
            sensivity.append(metrics.recall_score(y_test, y_pred))  # Sensitivity = Recall
            specificity.append(metrics.recall_score(y_test, y_pred))  # Note: This should be actual specificity calculation
            F1.append(metrics.f1_score(y_test, y_pred))

        # Return mean and standard deviation of all metrics
        return (
            np.mean(accuracy), np.std(accuracy),
            np.mean(precision), np.std(precision), 
            np.mean(sensivity), np.std(sensivity),
            np.mean(specificity), np.std(specificity),
            np.mean(F1), np.std(F1) )