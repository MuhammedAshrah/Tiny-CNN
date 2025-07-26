import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml

def load_mnist(batch_size=64):
    """
    Loads MNIST dataset from OpenML, processes it for PyTorch, and returns DataLoaders.

    args :
        batch_size: Number of samples per batch

    Returns:
        train_loader: DataLoader yielding (images, labels) for training
        test_loader: DataLoader yielding (images, labels) for testing
    """
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, parser="auto")

    # First step is to split the dataset into train and test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # normalize pixel values to [0, 1]
    X_train = np.array(X_train) / 255.0
    X_test = np.array(X_test) / 255.0
    y_train=np.array(y_train,dtype=np.int8)
    y_test= np.array(y_test, dtype=np.int8)

    # Reshape from (N, 784) to (N, 1, 28, 28)
    # This is because pytorch wants us to have data in the shape(batch_size,channel,height,width)
    # -1 here allows the batch size to be inferred automatically
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    # Converting to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.astype(np.int64), dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.astype(np.int64), dtype=torch.long)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create loaders
    train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader= DataLoader(test_dataset, batch_size=batch_size)

    print("MNIST loaded successfully.")
    return train_loader, test_loader
