import numpy as np
import torch
from typing import List, Optional, Tuple

class TabularDataset(torch.utils.data.Dataset):
    """
    Dataset that can be passed to nntd.pipelines.Pipeline for training.
    Attributes:
        X: torch.Tensor of features
        y: torch.Tensor of targets
        n_num_features: number of numeric columns of X
        n_cat_features: number of categorical columns of X
        cardinalities: list of number of unique classes for each categorical
        storage_device: torch.device to store tensors X and y
    """
    def __init__(
            self,
            X_num: np.ndarray,
            X_cat: Optional[np.ndarray],
            y: np.ndarray,
            cardinalities: List[int],
            device: torch.device,
    ) -> None:
        """
        Args:
            X_num: array-like of numeric features
            X_cat: array-like of categorical features; values should be ints from 0 to n_classes-1
        X_num and X_cat will be concatenated together to create tensor X on the storage_device
        """
        assert (
            X_cat is None or len(cardinalities) == X_cat.shape[1]
        ), 'len(cardinalities) must match the number of columns of X_cat if X_cat is not None'
        self.X = torch.tensor(
            X_num if not cardinalities else np.concatenate((X_num, X_cat), axis=1),
            dtype=torch.float,
            device=device,
            )
        self.y = torch.tensor(y, dtype=torch.float, device=device)
        self.n_num_features = X_num.shape[-1]
        self.n_cat_features = self.X.shape[-1]-self.n_num_features
        self.cardinalities = cardinalities
        self.storage_device = device
    
    def __getitem__(self, idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def __len__(self) -> int:
        return len(self.y)
    
    def get_numeric_features(self) -> torch.Tensor:
        return self.X[:, :self.n_num_features]
    
    def get_categorical_features(self) -> torch.Tensor:
        return self.X[:, self.n_num_features:]

    def change_storage_device(self, device: torch.device) -> None:
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        self.storage_device = device
