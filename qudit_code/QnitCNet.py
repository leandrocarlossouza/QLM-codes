import joblib
from enum import Enum
import numpy as np
from scipy.special import expit
from scipy.linalg import lstsq
from scipy.sparse.linalg import cg
from itertools import combinations_with_replacement
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss
from joblib import Parallel, delayed
import os
import psutil

class QnitCNet():
    """
    Quantum-inspired Neural Network Classifier (QnitCNet) for multi-class classification.
    
    This classifier combines polynomial feature expansion with a cascade of linear SVMs,
    using a quantum-inspired decision mechanism based on trigonometric functions of the 
    SVM outputs. The model is particularly suited for problems where feature interactions
    are important.
    
    Parameters:
    -----------
    xt : np.array
        Training data matrix of shape (n_samples, n_features)
    yt : np.array
        Training labels vector of shape (n_samples,)
    neuron_number : int
        Determines the complexity of feature transformations:
        - For power_type=0: maximum polynomial degree
        - For power_type=1: maximum interaction terms
    power_type : int, optional (default=0)
        Type of feature preprocessing:
        - 0: Polynomial features (x, x², x³,...)
        - 1: All interaction terms (x, xy, xyz,...)
    
    Attributes:
    -----------
    alphas : list
        Learned weight vectors for each binary classifier
    class_order : list
        Order in which classes were learned during training
    angles_number : int
        Number of distinct classes in the training data
    """
    
    def __init__(self, xt, yt, neuron_number, power_type = 0):
        """
        Initializes the classifier with training data and parameters.
        
        Args:
            xt (np.array): Training data matrix (n_samples × n_features)
            yt (np.array): Training labels vector (n_samples,)
            neuron_number (int): Controls model complexity (see class docstring)
            power_type (int): Feature preprocessing type (0 or 1)
        """
        self.xt = xt
        self.yt = yt
        self.neuron_number = np.maximum(neuron_number, 1)  # Ensure at least one neuron
        self.nt = np.size(xt, 0)  # Number of training samples
        self.p = np.size(xt, 1)  # Number of features
        self.alphas = []  # Learned weight vectors for each binary classifier
        self.angles_number = np.max(yt)  # Number of classes (assuming 0-based)
        self.power_type = power_type  # Feature preprocessing type
        self.class_order = []  # Order in which classes are learned
        

    def train(self):
        """
        Train the classifier using parallel processing.
        
        This method:
        1. Preprocesses features based on power_type
        2. Trains binary classifiers for each class in parallel
        3. Selects optimal class order based on hinge loss
        
        Returns:
            int: Number of features after preprocessing
        """
        # Preprocess the entire dataset
        if self.power_type == 0:
            mx = self.preprocess_dataP(self.xt)
        else:
            mx = self.preprocess_dataM(self.xt)

        def train_class(target_class):
            """
            Inner function for training a binary classifier for one class.
            
            Args:
                target_class (int): Class to train against all others
                
            Returns:
                tuple: (target_class, hinge_loss, weight_vector)
            """
            # Create mask excluding already classified samples
            mask = self.yt >= 0
            for k in self.class_order:
                mask &= (self.yt != k)
            nx = mx[mask,:]
            ny = self.yt[mask]

            # Create binary labels (-1 for target class, 1 for others)
            ny = np.where(ny == target_class, -1, 1)
            
            # Skip if only one class remains
            if(sum(ny==-1)==np.size(nx,0)):
                return target_class, 0, 0

            # Train linear SVM
            model = LinearSVC(C=100, max_iter=100, tol=0.01)
            model.fit(nx, ny)

            # Create weight vector (scaled by 100 for numerical stability)
            alpha = 100 * np.concatenate([model.intercept_, model.coef_.flatten()])

            # Calculate hinge loss
            y_pred = model.predict(nx)
            erro = hinge_loss(ny, y_pred)

            return target_class, erro, alpha


        # Parallel training across all classes
        for z in reversed(range(0, self.angles_number + 1)):
            best_error = 1e100
            selected_class = []
            best_alpha = []

            # Parallel execution across all remaining classes
            results = Parallel(n_jobs=-1)(
                delayed(train_class)(target_class)
                for target_class in reversed(range(0, self.angles_number + 1))
                if target_class not in self.class_order
            )

            # Process parallel results
            for target_class, error, alpha in results:
                if len(self.class_order) == self.angles_number:
                    selected_class = target_class
                    best_alpha = alpha
                    break

                if error < best_error:
                    best_error = error
                    selected_class = target_class
                    best_alpha = alpha

            self.class_order.append(selected_class)
            self.alphas.append(best_alpha)

        return (np.size(mx, 1)+1)

    
    def trainNP(self):
        """
        Non-parallel version of train() for debugging or comparison.
        
        Returns:
            int: Number of features after preprocessing
        """
        self.class_order = []
        nv  =[]
        if self.power_type == 0:
            mx = self.preprocess_dataP(self.xt)
        else:
            mx = self.preprocess_dataM(self.xt)
            
        for z in reversed(range(0,self.angles_number+1)):
            best_error = 1e100
            selected_class = []
            best_alpha = []
            
            for target_class in reversed(range(0,self.angles_number+1)):
                if target_class in self.class_order:
                    continue

                if(len(self.class_order) == self.angles_number):
                    selected_class = target_class
                    best_alpha = alpha
                    break

                # Create mask excluding already classified samples
                mask = self.yt >= 0
                for k in self.class_order:
                    mask &= (self.yt != k)

                nx = mx[mask,:]
                ny = self.yt[mask]
    
                # Create binary labels
                ny = np.where(ny == target_class, -1, 1)

                # Train linear SVM
                model = LinearSVC(C=1, max_iter=100, tol = 0.01)
                model.fit(nx[:, 1:], ny)
        
                # Create weight vector
                alpha = 100*np.concatenate([model.intercept_, model.coef_.flatten()])

                # Calculate hinge loss
                y_pred = model.predict(nx[:, 1:])                
                erro = hinge_loss(ny, y_pred)

                if(erro < best_error):
                    best_error = erro 
                    selected_class = target_class
                    best_alpha = alpha

            self.class_order.append(selected_class)
            self.alphas.append(best_alpha)
            
        return (np.size(mx, 1)+1)
    

    def predict(self, sx):
        """
        Predict class labels for new samples using quantum-inspired decision rule.
        
        The decision rule uses trigonometric functions of the linear classifier outputs:
        1. Converts SVM outputs to angles using arcsin(sigmoid(output))
        2. Computes quantum-inspired probabilities using products of sines
        3. Selects class with maximum "quantum probability"
        
        Args:
            sx (np.array): Test samples (n_samples × n_features)
            
        Returns:
            np.array: Predicted class labels
        """
        if self.power_type == 0:
            nx = self.preprocess_dataP(sx)
        else:
            nx =self.preprocess_dataM(sx)

        # Add bias term
        nx = np.concatenate((np.ones((np.size(sx, 0), 1)), nx), axis=1)

        predictions = np.zeros(nx.shape[0], dtype=int)

        for i in range(nx.shape[0]):
            # Convert classifier outputs to angles
            alpha = np.zeros(self.angles_number)
            for z in range(0, self.angles_number):
                alpha[z] = np.arcsin(expit(np.dot(nx[i], self.alphas[z])))

            # Initialize with "all sines" probability
            hk = np.prod(np.sin(alpha[:]))**2 
            predictions[i] = self.class_order[self.angles_number]

            # Compare with partial products
            for k in range(1, self.angles_number,1):
                ok = (np.prod(np.sin(alpha[:k])) * np.cos(alpha[k]))**2
                if ok > hk:
                    hk = ok
                    predictions[i] = self.class_order[k]

            # Check pure cosine probability
            ok = (np.cos(alpha[0]))**2
            if ok > hk:
                hk = ok
                predictions[i] = self.class_order[0]

        return predictions


    def preprocess_dataP(self, x):
        """
        Polynomial feature preprocessing (power_type=0).
        
        Generates polynomial features up to degree neuron_number:
        [x, x², x³, ..., x^neuron_number]
        
        Args:
            x (np.array): Input data (n_samples × n_features)
            
        Returns:
            np.array: Augmented feature matrix
        """
        n_samples, n_features = x.shape
        r = x  # Start with original features
        
        # Add polynomial terms
        for k in range(2, self.neuron_number + 1):
            r = np.concatenate((r, x ** k), axis=1)
        
        return r.astype(np.float32)


    def preprocess_dataM(self, x):
        """
        Interaction term preprocessing (power_type=1).
        
        Generates all interaction terms up to order neuron_number:
        [x, xy, xyz, ...] for all feature combinations.
        
        Args:
            x (np.array): Input data (n_samples × n_features)
            
        Returns:
            np.array: Augmented feature matrix
        """
        n_samples, n_features = x.shape
    
        # Start with original features
        r = [x]  
        
        # Add interaction terms
        for k in range(2,self.neuron_number + 1, 1):
            combs = list(combinations_with_replacement(range(n_features), k))
            for comb in combs:
                r.append(np.prod(x[:, comb], axis=1).reshape(-1, 1))
          
        # Combine all features
        r = np.hstack(r)
    
        return r.astype(np.float32)