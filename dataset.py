import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple

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

class NaNAugmentation(nn.Module):
    """
    Module that adds NaNs to a batch of features
    Attributes:
        p_nan: probability of replacing numeric feature with NaN, or list of values for every numeric feature
        n_num_features: number of numeric features in the dataset
    """
    def __init__(self, p_nan: Union[List[float], float], n_num_features: int, init_device: torch.device) -> None:
        if isinstance(p_nan, float):
            p_nan = [p_nan]*n_num_features
        assert(
            len(p_nan) == n_num_features
        ), 'len(p_nan) must equal the number of features if a list is passed'
        super().__init__()
        p_nan = torch.tensor([p_nan], device=init_device)
        self.register_buffer('p_nan', p_nan)
        self.n_num_features = n_num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            device = self.p_nan.device
            mask = torch.rand((len(x), self.n_num_features), device=device) < self.p_nan
            mask = torch.cat(
                (
                    mask,
                    torch.zeros((len(x), x.shape[-1]-self.n_num_features), dtype=torch.bool, device=device)
                ),
                dim=1
            )#pad the mask to the size of x to allow for categorical features
            x = torch.where(mask, np.nan, x)
        return x

class GaussianNoise(nn.Module):
    """
    Module that adds Gaussian noise to numeric data
        sigma: noise standard deviation, or list of values for every numeric feature
        n_num_features: number of numeric features in the dataset
    """
    def __init__(self, sigma: Union[List[float], float], n_num_features: int, init_device: torch.device) -> None:
        if isinstance(sigma, float):
            sigma = [sigma]*n_num_features
        assert(
            len(sigma) == n_num_features
        ), 'len(sigma) must equal the number of features if a list is passed'
        super().__init__()
        sigma = torch.tensor([sigma], device=init_device)
        self.register_buffer('sigma', sigma)
        self.n_num_features = n_num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            device = self.sigma.device
            noise = self.sigma*torch.normal(0, 1, x.shape, device=device)
            noise[:, self.n_num_features:] = 0
            x = x+noise
        return x

class AugmentedTabularDataset(TabularDataset):
    """
    Version of TabularDataset that includes data augmentation for numeric features
    Attributes:
        nan_augmentation: module that randomly adds NaNs to numeric data
        gaussian_noise: module that adds gaussian noise
    """
    def __init__(
            self,
            X_num: np.ndarray,
            X_cat: Optional[np.ndarray],
            y: np.ndarray,
            cardinalities: List[int],
            device: torch.device,
            p_nan: Optional[Union[List[float], float]] = None,
            sigma: Optional[Union[List[float], float]] = None,
    ) -> None:
        """
        Args:
            p_nan: probability of replacing numeric feature with NaN, or list of values for every numeric feature
            sigma: standard deviation of gaussian noise, or list of values for every numeric feature
        """
        super().__init__(X_num, X_cat, y, cardinalities, device)
        self.nan_augmentation = None if p_nan is None else NaNAugmentation(p_nan, self.n_num_features, device)
        self.gaussian_noise = None if sigma is None else GaussianNoise(sigma, self.n_num_features, device)

    def change_storage_device(self, device: torch.device) -> None:
        super().change_storage_device(device)
        if self.nan_augmentation is not None:
            self.nan_augmentation.to(device)
        if self.gaussian_noise is not None:
            self.gaussian_noise.to(device)

    def __getitem__(self, idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = super().__getitem__(idx)
        if self.nan_augmentation is not None:
            x = self.nan_augmentation(x)
        if self.gaussian_noise is not None:
            x = self.gaussian_noise(x)
        return x, y
