"""
SQQNN Implementation
This class implements a quantum-inspired neural network using the QNeuron as building blocks.
It supports both regression and classification tasks with different loss functions.
"""

import numpy as np
import joblib      
from QNeuron import *
from enum import Enum
import random

class LOSS_FUNCTION(Enum):
    """Enumeration of supported loss functions"""
    HINGE_LOSS = 1    # For classification tasks
    MEAN_SQUARED = 2  # For regression tasks

class QNetwork():    
    """
    Quantum-inspired neural network that can be trained using different optimization methods.
    
    Attributes:
        xt, xv: Training and validation feature sets
        yt, yv: Training and validation targets (normalized)
        neuron_set: Collection of QNeurons
        training_errors: History of training errors
        validation_errors: History of validation errors
    """
    
    def __init__(self, xt, yt, xv, yv, neuron_number):        
        """
        Initialize the quantum network with data.
        
        Args:
            xt: Training features (n_samples x n_features)
            yt: Training targets
            xv: Validation features
            yv: Validation targets
            neuron_number: Number of neurons/polynomial degree for preprocessing
        """
        self.xt = xt  # Training features
        self.xv = xv  # Validation features
        self.training_errors = []  # To track training error over iterations
        self.validation_errors = []  # To track validation error over iterations
        self.nt = np.size(xt,0)  # Number of training samples
        self.nv = np.size(xv,0)  # Number of validation samples
        self.p = np.size(xt,1)  # Number of features
        self.neuron_set = []  # Will hold our QNeuron(s)
        self.neuron_number = np.maximum(neuron_number,1)  # Ensure at least 1 neuron
        
        # Normalize targets to [0,1] range
        self.my = np.min(yt)  # Minimum target value
        self.My = np.max(yt)  # Maximum target value
        self.yt = ((yt-self.my)/(self.My-self.my))  # Normalized training targets
        self.yv = ((yv-self.my)/(self.My-self.my))  # Normalized validation targets
        
        # Placeholders for later use
        self.pH = None  # Will store pseudo-inverse matrix
        self.nxt = None  # Will store preprocessed training data

    def get_errors(self):
        """Return training and validation error histories"""
        return self.training_errors, self.validation_errors

    def predict(self, x):
        """
        Make predictions on new data.
        
        Args:
            x: Input features to predict on
            
        Returns:
            Predictions in original target scale
        """
        nx = self.preprocess_data(x)  # Apply same preprocessing
        # Predict and scale back to original range
        return (self.neuron_set.predict(nx))*(self.My-self.my)+self.my

    def compute_error(self, yh, y, n, loss_function):
        """
        Compute error between predictions and targets.
        
        Args:
            yh: Predicted values
            y: True values
            n: Number of samples
            loss_function: Which error metric to use
            
        Returns:
            Computed error value
        """
        if loss_function == LOSS_FUNCTION.HINGE_LOSS:
            # Hinge loss for classification
            return np.sum(np.maximum(1 - yh*(2*y-1), 0))/n
        else:
            # MSE for regression
            diff = yh-y
            return np.dot(diff.T, diff)/n

    def tanhlinear_regression(self, scale):
        """
        Fit using tanh-linear regression approach.
        
        Args:
            scale: Scaling factor for target transformation
            
        Returns:
            Computed error
        """
        a = -1+scale
        b = 1-scale 
        # Use ARC_TANH_L activation for all angles
        rt = np.array([ATIVATION_TYPE.ARC_TANH_L]*5)
        self.neuron_set = QNeuron(self.p, rt, 1)  # linear_reg=1
        
        # Transform targets
        yj = self.yt*(b-a)+a        
        yj = np.squeeze(np.arctanh(yj))
        
        # Solve linear system
        self.neuron_set.alphas = np.squeeze(np.dot(self.pH, yj))
        
        # Compute and return error
        ya = self.predict(self.xt)
        return self.compute_error(ya, np.squeeze(self.yt).T, self.nt, LOSS_FUNCTION.HINGE_LOSS)
            
    def fit_regression(self, scale=1e-3):
        """
        Fit the model using regression approach.
        
        Args:
            scale: Small scaling factor
            
        Returns:
            Computed error
        """
        if self.nxt is None:
            self.nxt = self.preprocess_data(self.xt)
            
        Xt = np.transpose(self.nxt)
        if self.pH is None:
            # Compute pseudo-inverse
            self.pH = np.matmul(np.linalg.pinv(np.matmul(Xt, self.nxt)), Xt)
            
        Xt = []  # Free memory
        error = self.tanhlinear_regression(scale)
        
        # Clean up
        self.pH = []
        self.nxt = []
        return error
        
    def preprocess_data(self, x):
        """
        Preprocess data by adding polynomial features.
        
        Args:
            x: Input features
            
        Returns:
            Augmented feature matrix with polynomial terms
        """
        r = np.ones((np.size(x,0),1))  # Start with bias term
        for k in range(0, self.neuron_number+1):
            r = np.concatenate((r, x**(k+1)), axis=1)
        return r
    
    def fit_gradient(self, loss_function=LOSS_FUNCTION.MEAN_SQUARED, n_iterations=100, 
                    pct=0.1, lr=0.1, initialize=True):
        """
        Main training method that routes to specific gradient descent implementation.
        """
        if loss_function == LOSS_FUNCTION.MEAN_SQUARED:
            return self.fit_gradient_descent_MEAN_SQUARED(n_iterations, pct, lr, 0.9)
        else:
            return self.fit_gradient_descent_HINGE_LOSS(n_iterations, pct, lr, 0.9)

    def fit_gradient_descent_MEAN_SQUARED(self, n_iterations=100, pct=0.1, lr=0.1, 
                                        grad_update=0.1, initialize=True, tol=1e-6):
        """
        Train using gradient descent with mean squared error loss.
        
        Args:
            n_iterations: Max number of iterations
            pct: Percentage of samples per mini-batch
            lr: Learning rate
            grad_update: Gradient update parameter
            initialize: Whether to initialize new parameters
            tol: Tolerance for early stopping
            
        Returns:
            Final training error, number of iterations
        """
        # Use linear activation for all angles
        rt = np.array([ATIVATION_TYPE.LINEAR]*5)
        
        # Preprocess data
        nxt = self.preprocess_data(self.xt)
        nxv = self.preprocess_data(self.xv)
        pd = np.size(nxt,1)  # Dimension of preprocessed data
        nbatch = int(pct*self.nt)  # Batch size
        
        # Initialize or reuse neuron parameters
        if initialize or not self.neuron_set:
            neuron = QNeuron(self.p, rt)
            # Random initialization
            neuron.alphas = np.random.random(pd).T
            neuron.bethas = np.random.random(pd).T
            neuron.gammas = np.random.random(pd).T
            neuron.omegas = np.random.random(pd).T     
            neuron.thetas = np.random.random(pd).T
        else:
            neuron = self.neuron_set
            
        self.grad_squared = 0  # For adaptive learning rate

        def update_with_gradient():
            """Compute gradients and update parameters for current batch"""
            # Initialize gradient accumulators
            dh_alphas = np.zeros(pd)
            dh_bethas = np.zeros(pd)
            dh_gammas = np.zeros(pd)
            dh_omegas = np.zeros(pd)
            dh_thetas = np.zeros(pd)
            
            # Sample mini-batch
            indexes = random.sample(range(0,self.nt), nbatch)
            x_data = nxt[indexes,:]
            y_data = np.squeeze(self.yt[indexes]).T
            
            # Compute all angles and their trig functions
            alpha = neuron.get_transformed_angles(np.dot(x_data, neuron.alphas), rt[0])
            beta = neuron.get_transformed_angles(np.dot(x_data, neuron.bethas), rt[1])   
            gamma = neuron.get_transformed_angles(np.dot(x_data, neuron.gammas), rt[2]) 
            omega = neuron.get_transformed_angles(np.dot(x_data, neuron.omegas), rt[3]) 
            theta = neuron.get_transformed_angles(np.dot(x_data, neuron.thetas), rt[4]) 
            
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
            
            # Compute predictions
            yh = (sin_theta * sin_omega * cos_gamma * cos_beta * cos_alpha +
                  sin_omega * sin_beta * cos_theta * cos_gamma +
                  sin_theta * sin_beta * cos_omega * cos_alpha -
                  sin_theta * sin_omega * sin_gamma * sin_alpha - 
                  cos_beta * cos_omega * cos_theta)
            
            # Compute gradients for each parameter
            # Alpha gradient
            d_alpha = (-sin_theta * sin_omega * cos_gamma * cos_beta * sin_alpha -
                       sin_theta * sin_beta * cos_omega * sin_alpha -            
                       sin_theta * sin_omega * sin_gamma * cos_alpha)
            ct = (yh-y_data)*d_alpha
            dh_alphas = np.squeeze(np.dot(ct.T, x_data)).T

            # Beta gradient
            d_beta = (-sin_theta * sin_omega * cos_gamma * sin_beta * cos_alpha + 
                      sin_omega * cos_beta * cos_theta * cos_gamma +
                      sin_theta * cos_beta * cos_omega * cos_alpha + 
                      sin_beta * cos_omega * cos_theta)
            ct = (yh-y_data)*d_beta
            dh_bethas = np.squeeze(np.dot(ct.T, x_data)).T
     
            # Gamma gradient
            d_gamma = (-sin_theta * sin_omega * sin_gamma * cos_beta * cos_alpha -
                       sin_omega * sin_beta * cos_theta * sin_gamma -
                       sin_theta * sin_omega * cos_gamma * sin_alpha)
            ct = (yh-y_data)*d_gamma
            dh_gammas = np.squeeze(np.dot(ct.T, x_data)).T           
                  
            # Omega gradient
            d_omega = (sin_theta * cos_omega * cos_gamma * cos_beta * cos_alpha +
                       cos_omega * sin_beta * cos_theta * cos_gamma -
                       sin_theta * sin_beta * sin_omega * cos_alpha -
                       sin_theta * cos_omega * sin_gamma * sin_alpha +
                       cos_beta * sin_omega * cos_theta)    
            ct = (yh-y_data)*d_omega
            dh_omegas = np.squeeze(np.dot(ct.T, x_data)).T    
    
            # Theta gradient
            d_theta = (cos_theta * sin_omega * cos_gamma * cos_beta * cos_alpha -
                       sin_omega * sin_beta * sin_theta * cos_gamma +
                       cos_theta * sin_beta * cos_omega * cos_alpha -
                       cos_theta * sin_omega * sin_gamma * sin_alpha + 
                       cos_beta * cos_omega * sin_theta)
            ct = (yh-y_data)*d_theta
            dh_thetas = np.squeeze(np.dot(ct.T, x_data))

            # Update parameters with adaptive learning rate
            sg = sum(dh_alphas**2)+sum(dh_bethas**2)+sum(dh_omegas**2)+sum(dh_gammas**2)+sum(dh_thetas**2)
            self.grad_squared = (1-grad_update)*self.grad_squared + grad_update*sg
            lr_adjusted = lr/np.sqrt(self.grad_squared)
            
            neuron.alphas -= lr_adjusted*dh_alphas/nbatch
            neuron.bethas -= lr_adjusted*dh_bethas/nbatch
            neuron.gammas -= lr_adjusted*dh_gammas/nbatch
            neuron.omegas -= lr_adjusted*dh_omegas/nbatch       
            neuron.thetas -= lr_adjusted*dh_thetas/nbatch
            
        # Training loop
        last_error = 1e100  # Initialize with large value
        self.neuron_set = neuron
        last_pct = 0
        itr_number = 0
        
        for r in range(n_iterations):
            itr_number += 1
            update_with_gradient()
            
            # Compute errors
            ya = neuron.predict(nxv)
            yz = neuron.predict(nxt)
            cost = self.compute_error(ya, np.squeeze(self.yv).T, self.nv, LOSS_FUNCTION.MEAN_SQUARED)
            costz = self.compute_error(yz, np.squeeze(self.yt).T, self.nt, LOSS_FUNCTION.MEAN_SQUARED)
            
            # Track errors
            self.training_errors.append(costz)  
            self.validation_errors.append(cost)
            
            # Early stopping and model selection
            if cost < last_error:
                last_error = cost
                self.neuron_set = neuron  # Keep best model
            else:
                break
                
            # Progress reporting
            now_pct = int(((r+1)/n_iterations)*100)
            if now_pct != last_pct:
                print("", end="\r")
                print(now_pct, end="% ", flush=True)
                last_pct = now_pct

            # Convergence check
            if costz < tol:
                break
               
        print("", end="\r")  # Clear progress line

        # Return final training error and iteration count
        ya = self.neuron_set.predict(nxt)
        cost = self.compute_error(ya, np.squeeze(self.yt).T, self.nt, LOSS_FUNCTION.MEAN_SQUARED)
        return cost, itr_number
    
    def fit_gradient_descent_HINGE_LOSS(self, n_iterations=100, pct=0.1, lr=0.1, grad_update=0.1):
        """
        Train using gradient descent with hinge loss (for classification).
        
        Args:
            n_iterations: Max number of iterations
            pct: Percentage of samples per mini-batch
            lr: Learning rate
            grad_update: Gradient update parameter
            
        Returns:
            Final training error
        """
        # Similar structure to MEAN_SQUARED version but with hinge loss
        rt = np.array([ATIVATION_TYPE.LINEAR]*5)
        nxt = self.preprocess_data(self.xt)
        nxv = self.preprocess_data(self.xv)
        pd = np.size(nxt,1)
        nbatch = int(pct*self.nt)        
        neuron = QNeuron(self.p, rt)
        
        # Random initialization
        neuron.alphas = np.random.random(pd).T
        neuron.bethas = np.random.random(pd).T
        neuron.gammas = np.random.random(pd).T
        neuron.omegas = np.random.random(pd).T     
        neuron.thetas = np.random.random(pd).T
        self.grad_squared = 0
        
        def sgn(yp, yo):
            """Helper function for hinge loss gradient"""
            return 1 - (2*yp-1)*(2*yo-1) > 0

        def update_with_gradient():
            """Compute gradients and update parameters for hinge loss"""
            # Similar to MEAN_SQUARED version but with hinge-specific gradient
            dh_alphas = np.zeros(pd)
            dh_bethas = np.zeros(pd)
            dh_gammas = np.zeros(pd)
            dh_omegas = np.zeros(pd)
            dh_thetas = np.zeros(pd)
            
            indexes = random.sample(range(0,self.nt), nbatch)
            x_data = nxt[indexes,:]
            y_data = np.squeeze(self.yt[indexes]).T
            
            # Compute all angles and trig functions (same as before)
            alpha = neuron.get_transformed_angles(np.dot(x_data, neuron.alphas), rt[0])
            beta = neuron.get_transformed_angles(np.dot(x_data, neuron.bethas), rt[1])   
            gamma = neuron.get_transformed_angles(np.dot(x_data, neuron.gammas), rt[2]) 
            omega = neuron.get_transformed_angles(np.dot(x_data, neuron.omegas), rt[3]) 
            theta = neuron.get_transformed_angles(np.dot(x_data, neuron.thetas), rt[4]) 
            
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
            
            # Compute predictions
            yh = (sin_theta * sin_omega * cos_gamma * cos_beta * cos_alpha +
                  sin_omega * sin_beta * cos_theta * cos_gamma +
                  sin_theta * sin_beta * cos_omega * cos_alpha -
                  sin_theta * sin_omega * sin_gamma * sin_alpha - 
                  cos_beta * cos_omega * cos_theta)
            
            # Hinge loss specific gradient computation
            ms = -1*sgn(yh, y_data)*(2*y_data-1)
            
            # Compute gradients (similar structure but with hinge loss factor)
            d_alpha = (-sin_theta * sin_omega * cos_gamma * cos_beta * sin_alpha -
                       sin_theta * sin_beta * cos_omega * sin_alpha -
                       sin_theta * sin_omega * np.sin(gamma) * cos_alpha)
            ct = ms*d_alpha
            dh_alphas = np.squeeze(np.dot(ct.T, x_data)).T

            d_beta = (-sin_theta * sin_omega * cos_gamma * sin_beta * cos_alpha + 
                      sin_omega * cos_beta * cos_theta * cos_gamma +
                      sin_theta * cos_beta * cos_omega * cos_alpha + 
                      sin_beta * cos_omega * cos_theta)
            ct = ms*d_beta
            dh_bethas = np.squeeze(np.dot(ct.T, x_data)).T
     
            d_gamma = (-sin_theta * sin_omega * sin_gamma * cos_beta * cos_alpha -
                       sin_omega * sin_beta * cos_theta * sin_gamma -
                       sin_theta * sin_omega * cos_gamma * sin_alpha)
            ct = ms*d_gamma
            dh_gammas = np.squeeze(np.dot(ct.T, x_data)).T           
                  
            d_omega = (sin_theta * cos_omega * cos_gamma * cos_beta * cos_alpha +
                       cos_omega * sin_beta * cos_theta * cos_gamma -
                       sin_theta * sin_beta * sin_omega * cos_alpha -
                       sin_theta * cos_omega * sin_gamma * sin_alpha +
                       cos_beta * sin_omega * cos_theta)    
            ct = ms*d_omega
            dh_omegas = np.squeeze(np.dot(ct.T, x_data)).T    
    
            d_theta = (cos_theta * sin_omega * cos_gamma * cos_beta * cos_alpha -
                       sin_omega * sin_beta * sin_theta * cos_gamma +
                       cos_theta * sin_beta * cos_omega * cos_alpha -
                       cos_theta * sin_omega * sin_gamma * sin_alpha + 
                       cos_beta * cos_omega * sin_theta)
            ct = ms*d_theta
            dh_thetas = np.squeeze(np.dot(ct.T, x_data))

            # Update parameters with adaptive learning rate
            sg = sum(dh_alphas**2)+sum(dh_bethas**2)+sum(dh_omegas**2)+sum(dh_gammas**2)+sum(dh_thetas**2)
            self.grad_squared = (1-grad_update)*self.grad_squared + grad_update*sg
            
            neuron.alphas -= (lr/np.sqrt(self.grad_squared))*dh_alphas/nbatch
            neuron.bethas -= (lr/np.sqrt(self.grad_squared))*dh_bethas/nbatch
            neuron.gammas -= (lr/np.sqrt(self.grad_squared))*dh_gammas/nbatch
            neuron.omegas -= (lr/np.sqrt(self.grad_squared))*dh_omegas/nbatch       
            neuron.thetas -= (lr/np.sqrt(self.grad_squared))*dh_thetas/nbatch
            
        # Training loop
        last_error = 1e100
        self.neuron_set = neuron
        last_pct = 0
        
        for r in range(n_iterations):
            update_with_gradient()
            
            # Compute errors
            ya = neuron.predict(nxv)
            yz = neuron.predict(nxt)
            cost = self.compute_error(ya, np.squeeze(self.yv).T, self.nv, LOSS_FUNCTION.HINGE_LOSS)
            costz = self.compute_error(yz, np.squeeze(self.yt).T, self.nt, LOSS_FUNCTION.HINGE_LOSS)
            
            # Track errors
            self.training_errors.append(costz)  
            self.validation_errors.append(cost)
            
            # Early stopping and model selection
            if cost < last_error:
                last_error = cost
                self.neuron_set = neuron
            else:
                break
                
            # Progress reporting
            now_pct = int(((r+1)/n_iterations)*100)
            if now_pct != last_pct:
                print("", end="\r")
                print(now_pct, end="% ", flush=True)
                last_pct = now_pct
               
        print("", end="\r")  # Clear progress line

        # Return final training error
        ya = self.neuron_set.predict(nxt)
        return self.compute_error(ya, np.squeeze(self.yt).T, self.nt, LOSS_FUNCTION.HINGE_LOSS)