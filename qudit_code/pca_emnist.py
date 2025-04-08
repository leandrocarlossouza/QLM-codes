import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Dataset details
    print(f"{split.upper()} - Training samples: {len(train_dataset)}")
    print(f"{split.upper()} - Test samples: {len(test_dataset)}")
    print(f"{split.upper()} - Number of classes: {len(train_dataset.classes)}")
    
    return train_loader, test_loader

def fit_pca_on_full_dataset(set_name):
    """
    Fit PCA on the entire dataset (train + test data) and transform the images.
    
    Args:
        set_name: The name of the EMNIST dataset split (e.g., "balanced", "byclass", "digits", etc.).
        n_components: Number of principal components to retain.
    
    Returns:
        pca_data: PCA-transformed data (both train and test data combined).
        labels: Corresponding labels for the combined data (train and test).
    """
    # Load the train and test loaders using the specified dataset split
    train_loader, test_loader = load_emnist_split(set_name)

    # Collect all images and labels from the train and test loaders
    all_images = []
    all_labels = []

    # Collect images and labels from the training set
    for images, labels in train_loader:
        all_images.append(images.view(images.size(0), -1))  # Flatten images to 1D
        all_labels.append(labels)  # Collect corresponding labels

    # Collect images and labels from the test set
    for images, labels in test_loader:
        all_images.append(images.view(images.size(0), -1))  # Flatten images to 1D
        all_labels.append(labels)  # Collect corresponding labels

    # Stack all the flattened images into one array
    all_images = torch.cat(all_images, dim=0).numpy()  # Convert to NumPy
    all_labels = torch.cat(all_labels, dim=0).numpy()  # Convert to NumPy

    # Return the PCA-transformed data and corresponding labels as separate variables
    return all_images, all_labels  # Separate PCA data and labels




def process_image_with_pca(image, label, pca):
    """
    Process a single image: Apply PCA transformation and return the coefficients and label.
    """
    image_flattened = image.view(1, -1).numpy()  # Flatten image
    pca_coefficients = pca.transform(image_flattened)  # Apply PCA transformation
    return pca_coefficients.flatten(), label.item()  # Convert label to Python scalar

def parallel_process_with_pca(train_loader, pca, num_workers=-1):
    """
    Parallelize the processing of batches in the train_loader using pre-trained PCA.
    
    Args:
        train_loader: DataLoader object for loading batches of data.
        pca: Trained PCA model.
        num_workers: Number of threads to use for parallel processing.
    
    Returns:
        xt: NumPy array of processed PCA coefficients.
        yt: NumPy array of corresponding labels.
    """
    xt = []
    yt = []

    for images, labels in train_loader:
        # Parallelize processing of individual images in the batch
        results = Parallel(n_jobs=num_workers)(
            delayed(process_image_with_pca)(images[i], labels[i], pca) 
            for i in range(len(images))
        )
        
        # Collect results
        for pca_coeffs, label in results:
            xt.append(pca_coeffs)
            yt.append(label)

    return np.array(xt), np.array(yt)
