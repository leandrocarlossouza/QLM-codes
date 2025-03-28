"""
Quantum-Inspired Neural Network Implementation
This module implements a quantum-inspired neuron using rotation angles and trigonometric operations.
The design mimics quantum computing principles for machine learning tasks.
"""

# Import required libraries
import joblib  # For parallel processing to speed up computations
from enum import Enum  # For defining activation types as enumerations
import numpy as np  # For numerical operations and array handling
# Note: 'import numpy as math' is redundant since numpy is already imported as np

class ATIVATION_TYPE(Enum):
    """
    Enumeration class defining the activation types for the quantum neuron.
    Note: 'ATIVATION_TYPE' appears to be a typo - should likely be 'ACTIVATION_TYPE'
    """
    LINEAR = 0  # Standard linear activation (no transformation)
    ARC_TANH_L = 1  # Arccos-tanh activation for quantum-inspired non-linearity

class QNeuron():
    """
    Quantum-inspired neuron class that implements rotation angle-based computations.
    The neuron can operate in either full quantum mode or simplified linear regression mode.
    """
    
    def __init__(self, p, rtypes, linear_reg=0):
        """
        Initialize the quantum neuron with given parameters.
        
        Args:
            p (int): Number of input features/dimensions
            rtypes (list): List of ATIVATION_TYPE values for each angle [alpha, beta, gamma, omega, theta]
            linear_reg (int, optional): Flag to use simplified linear regression mode (1) or full quantum mode (0)
        """
        self.p = p  # Number of input features
        self.rtypes = rtypes  # Activation types for each angle
        # Initialize empty lists for angle parameters (to be populated later)
        self.alphas = []  # Alpha rotation angles
        self.bethas = []  # Beta rotation angles (note: typically spelled 'betas')
        self.gammas = []  # Gamma rotation angles
        self.omegas = []  # Omega rotation angles
        self.thetas = []  # Theta rotation angles
        self.linear_reg = linear_reg  # Operation mode flag

    def get_all_angles_linear(self, xt):
        """
        Compute all five rotation angles for each input sample using linear combinations.
        Utilizes parallel processing for efficient computation.
        
        Args:
            xt (numpy.ndarray): Input feature matrix of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Matrix of computed angles of shape (n_samples, 5)
                          Columns represent [alpha, beta, gamma, omega, theta] angles
        """
        n = np.size(xt, 0)  # Number of samples
        angles = np.zeros((n, 5))  # Initialize output matrix
        
        # Define helper function for parallel processing
        def process_func(i):
            """Compute linear combination for a single sample"""
            return coef[0] + np.sum(coef[1:]*xt[i,:])
        
        # Compute each angle in parallel using joblib
        # Alpha angles
        coef = self.alphas
        r1 = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_func)(i) for i in range(n))
        angles[:, 0] = np.array(r1).T
        
        # Beta angles
        coef = self.bethas
        r1 = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_func)(i) for i in range(n))  
        angles[:, 1] = np.array(r1).T
        
        # Gamma angles
        coef = self.gammas
        r1 = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_func)(i) for i in range(n))
        angles[:, 2] = np.array(r1).T
        
        # Omega angles
        coef = self.omegas
        r1 = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_func)(i) for i in range(n))   
        angles[:, 3] = np.array(r1).T
        
        # Theta angles
        coef = self.thetas
        r1 = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_func)(i) for i in range(n))
        angles[:, 4] = np.array(r1).T
        
        return angles
        
    def get_transformed_angles(self, angles, rtype):
        """
        Apply activation function to the computed angles.
        
        Args:
            angles (numpy.ndarray): Input angles to transform
            rtype (ATIVATION_TYPE): Activation type to apply
            
        Returns:
            numpy.ndarray: Transformed angles
        """
        if rtype == ATIVATION_TYPE.ARC_TANH_L:
            # Apply quantum-inspired non-linear transformation
            return np.arccos(np.tanh(angles))
        # Linear activation returns angles unchanged
        return angles
    
    def predict(self, sx):
        """
        Make predictions using quantum-inspired computations.
        
        Args:
            sx (numpy.ndarray): Input feature matrix of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predictions of shape (n_samples,)
        """
        # Compute transformed alpha angle
        alpha = self.get_transformed_angles(np.dot(sx, self.alphas), self.rtypes[0])
        
        if self.linear_reg == 1:
            # Simplified linear regression mode - just use cosine of alpha
            return np.squeeze(np.cos(alpha)).T
        else:
            # Full quantum mode - compute all angles and their trigonometric functions
            beta = self.get_transformed_angles(np.dot(sx, self.bethas), self.rtypes[1])   
            gamma = self.get_transformed_angles(np.dot(sx, self.gammas), self.rtypes[2]) 
            omega = self.get_transformed_angles(np.dot(sx, self.omegas), self.rtypes[3]) 
            theta = self.get_transformed_angles(np.dot(sx, self.thetas), self.rtypes[4]) 
            
            # Precompute all trigonometric functions for efficiency
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_omega = np.sin(omega)
            cos_omega = np.cos(omega)
            sin_gamma = np.sin(gamma)
            cos_gamma = np.cos(gamma)
            sin_alpha = np.sin(alpha)
            cos_alpha = np.cos(alpha)
            sin_beta = np.sin(beta)
            cos_beta = np.cos(beta)
            
            # Quantum-inspired prediction formula
            r = (sin_theta * sin_omega * cos_gamma * cos_beta * cos_alpha +
                 sin_omega * sin_beta * cos_theta * cos_gamma +
                 sin_theta * sin_beta * cos_omega * cos_alpha -
                 sin_theta * sin_omega * sin_gamma * sin_alpha -
                 cos_beta * cos_omega * cos_theta)
            
            return np.squeeze(r).T
