import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import math
import numpy as np
from joblib import Parallel, delayed

# Define EMNIST dataset splits
dataset_splits = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]


# Transformation pipeline: Normalize pixel values
transform = transforms.Compose([
    transforms.Grayscale(),         # Ensure grayscale images
    transforms.ToTensor(),          # Convert to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Function to load a specific split of EMNIST
def load_emnist_split(split):
    print(f"Loading EMNIST split: {split}")

    custom_dir = '../../../emnist_data'
    
    # Load training and test sets
    train_dataset = datasets.EMNIST(
        root=custom_dir, split=split, train=True, download=True, transform=transform
    )
    test_dataset = datasets.EMNIST(
        root=custom_dir, split=split, train=False, download=True, transform=transform
    )
    
    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Dataset details
    print(f"{split.upper()} - Training samples: {len(train_dataset)}")
    print(f"{split.upper()} - Test samples: {len(test_dataset)}")
    print(f"{split.upper()} - Number of classes: {len(train_dataset.classes)}")
    
    return train_loader, test_loader

def dct_2d_numpy(image):
    """
    Compute the 2D Discrete Cosine Transform (DCT) of an image using NumPy/Scipy.
    
    Args:
        image (np.ndarray): Input image as a 2D NumPy array (H, W).
    
    Returns:
        np.ndarray: DCT of the image as a 2D NumPy array.
    """
    # Apply DCT on rows, then on columns
    dct_rows = dct(image, type=2, norm='ortho', axis=0)
    dct_2d = dct(dct_rows, type=2, norm='ortho', axis=1)
    return dct_2d

def zigzag_order(matrix, max_elements=None):
    """
    Convert a 2D matrix into a 1D array following a zig-zag pattern with an optional limit on elements.
    
    Args:
        matrix (np.ndarray): 2D input matrix (H, W).
        max_elements (int, optional): Maximum number of elements to extract. If None, extract all.
    
    Returns:
        np.ndarray: 1D array of matrix elements in zig-zag order, limited to max_elements.
    """
    h, w = matrix.shape
    zigzag_indices = []
    
    for s in range(h + w - 1):  # Sum of row and column indices
        if s % 2 == 0:
            for i in range(s + 1):
                row, col = i, s - i
                if row < h and col < w:
                    zigzag_indices.append((row, col))
        else:
            for i in range(s + 1):
                row, col = s - i, i
                if row < h and col < w:
                    zigzag_indices.append((row, col))
    
    # Limit the number of elements
    if max_elements is not None:
        zigzag_indices = zigzag_indices[:max_elements]
    
    return np.array([matrix[row, col] for row, col in zigzag_indices])

def process_image(image, label, coef_number):
    """
    Process a single image: Compute DCT, apply Zig-Zag, and return coefficients and label.
    """
    dct_coefficients = dct_2d_numpy(image.squeeze(0).numpy()/255)  # Remove channel and convert to NumPy
    zigzag_coeffs = zigzag_order(dct_coefficients, coef_number)  # Apply Zig-Zag
    return zigzag_coeffs, label.item()  # Convert label to Python scalar

def parallel_process(train_loader, coef_number, num_workers=-1):
    """
    Parallelize the processing of batches in the train_loader.
    
    Args:
        train_loader: DataLoader object for loading batches of data.
        coef_number: Number of DCT coefficients to retain.
        num_workers: Number of threads to use for parallel processing.
    
    Returns:
        xt: NumPy array of processed DCT coefficients.
        yt: NumPy array of corresponding labels.
    """
    xt = []
    yt = []

    for images, labels in train_loader:
        # Parallelize processing of individual images in the batch
        results = Parallel(n_jobs=num_workers)(
            delayed(process_image)(images[i], labels[i], coef_number) 
            for i in range(len(images))
        )
        
        # Collect results
        for zigzag_coeffs, label in results:
            xt.append(zigzag_coeffs)
            yt.append(label)

    return np.array(xt), np.array(yt)
