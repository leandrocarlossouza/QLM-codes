# Import necessary libraries with explanatory comments
from sklearn.model_selection import KFold  # For k-fold cross-validation
from sklearn.metrics import accuracy_score  # For calculating classification accuracy
from sklearn import metrics  # For additional metrics if needed
from sklearn.ensemble import RandomForestClassifier  # Alternative model for comparison
import numpy as np  # Fundamental package for numerical computations
from QnitCNet import *  # Custom quantum-inspired neural network classifier
from time import perf_counter  # High-precision timing for performance measurement
from sklearn.decomposition import PCA  # For dimensionality reduction
from joblib import Parallel, delayed  # For parallel processing
import numpy as np  # Fundamental package for numerical computations
from sklearn.model_selection import KFold  # For k-fold cross-validation
from sklearn.decomposition import PCA  # For dimensionality reduction
from time import perf_counter  # High-precision timing for performance measurement
from sklearn.metrics import accuracy_score  # For calculating classification accuracy
import os  # For OS operations
import gc  # Garbage collection for memory management

class KFolderCV:
    """
    A k-fold cross-validation framework with support for:
    - Parallel processing
    - Dimensionality reduction (PCA)
    - Custom model evaluation
    - Performance timing
    
    Parameters:
    -----------
    n_splits : int, default=10
        Number of folds for cross-validation
    """
    
    def __init__(self, n_splits=10):
        """
        Initialize the k-fold cross-validator.
        
        Args:
            n_splits (int): Number of folds to create (default=10)
        """
        self.n_splits = n_splits

    def cross_validateP(self, X, y, neurons, n_components, model_type):
        """
        Parallelized k-fold cross-validation with PCA preprocessing.
        
        Args:
            X (np.array): Feature matrix (n_samples × n_features)
            y (np.array): Target labels (n_samples,)
            neurons (int): Number of neurons for QnitCNet
            n_components (int): Number of PCA components to keep
            model_type (int): Type of preprocessing for QnitCNet (0 or 1)
            
        Returns:
            tuple: (mean_accuracy, std_accuracy, mean_time, std_time, n_features)
        """
        # Initialize k-fold splitter with fixed random state for reproducibility
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # Initialize lists to store metrics across folds
        accuracy = []
        precision = []
        sensivity = []
        specificity = []
        F1 = []
        time = []
    
        def process_fold(train_index, test_index):
            """
            Inner function to process a single fold (used for parallel execution).
            
            Args:
                train_index (np.array): Indices for training data
                test_index (np.array): Indices for test data
                
            Returns:
                tuple: (accuracy, execution_time, n_features)
            """
            # Split data into train/test sets
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            # Apply PCA dimensionality reduction
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)  # Fit on train, transform both
            X_test = pca.transform(X_test)
    
            # Normalize features to [-1, 1] range
            m = np.min(X_train, axis=0)
            M = np.max(X_train, axis=0)
            X_train = 2 * (X_train - m) / (M - m) - 1
            X_test = 2 * (X_test - m) / (M - m) - 1
    
            # Train the model with timing
            start_time = perf_counter()
            model = QnitCNet(X_train, y_train, neurons, model_type)
            nv = model.train()  # nv = number of features after preprocessing
            end_time = perf_counter()
            gc.collect()  # Explicit garbage collection
            
            execution_time = end_time - start_time
    
            # Make predictions and calculate accuracy
            y_pred = model.predict(X_test)
    
            return accuracy_score(y_test, y_pred), execution_time, nv
    
        # Parallelize the k-fold processing using joblib
        results = Parallel(n_jobs=5)(
            delayed(process_fold)(train_index, test_index) 
            for train_index, test_index in kf.split(X)
        )
    
        # Unpack and aggregate results from all folds
        for result in results:
            acc, execution_time, nv = result
            accuracy.append(acc)
            time.append(execution_time)
    
        # Return summary statistics
        return (np.mean(accuracy), np.std(accuracy), 
                np.mean(time), np.std(time), nv)

    
    def cross_validate(self, X, y, neurons, n_components, model_type):
        """
        Sequential version of k-fold cross-validation (for debugging/comparison).
        
        Args:
            X (np.array): Feature matrix (n_samples × n_features)
            y (np.array): Target labels (n_samples,)
            neurons (int): Number of neurons for QnitCNet
            n_components (int): Number of PCA components to keep
            model_type (int): Type of preprocessing for QnitCNet (0 or 1)
            
        Returns:
            tuple: (mean_accuracy, std_accuracy, mean_time, std_time, n_features)
        """
        # Initialize k-fold splitter
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # Initialize metrics storage
        accuracy = []
        precision = []
        sensivity = []
        specificity = []
        F1 = []
        time = []

        counter = 1  # Fold counter for progress tracking
        
        # Process each fold sequentially
        for train_index, test_index in kf.split(X):
            # Split data
            X_train, X_test = X[train_index,:], X[test_index,:]
            y_train, y_test = y[train_index], y[test_index]

            # Apply PCA
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            # Normalize features
            m = np.min(X_train, axis=0)
            M = np.max(X_train, axis=0)
            X_train = 2*(X_train-m)/(M-m)-1
            X_test = 2*(X_test-m)/(M-m)-1

            # Initialize and train model
            model = QnitCNet(X_train, y_train, neurons, model_type)
            
            # Time the training process
            start_time = perf_counter()
            nv = model.train()  # Number of features after preprocessing
            end_time = perf_counter()
            gc.collect()  # Manual garbage collection
            
            execution_time = end_time - start_time
            time.append(execution_time)
            
            # Make predictions and calculate accuracy
            y_pred = model.predict(X_test)

            # Print fold progress
            print(f"{counter}/{self.n_splits} -> accuracy: {100*accuracy_score(y_test, y_pred):.2f}  time: {execution_time:.2f}")
            counter += 1
            
            # Store accuracy for this fold
            accuracy.append(accuracy_score(y_test, y_pred))

        # Return aggregated results
        return (np.mean(accuracy), np.std(accuracy), 
                np.mean(time), np.std(time), nv)