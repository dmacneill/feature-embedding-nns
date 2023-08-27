import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from .modules import FeatureEmbeddingModel, AugmentedFeatureEmbeddingModel
from .dataset import TabularDataset
from typing import Iterator, Dict, Optional, Callable, List, Union, Tuple

class ProgressTracker:
    """
    Simple class to hold training results + to pass evaluation callback to the Pipeline class
    Attributes:
        score_lookback: how many epochs to average score for mid-training display output
        val_dataset: nntd.dataset.TabularDataset for evaluation
        val_metric: function to calculate validation score = val_metric(yhat: np.ndarray, y: np.ndarray)
        verbose: print validation results
        losses: list of all validation loss values
        score: most recent value = losses[-1]
    """
    def __init__(
            self,
            score_lookback: int,
            val_dataset: TabularDataset,
            val_metric: Callable[[np.ndarray, np.ndarray], float],
            verbose: Optional = True
    ) -> None:
        self.score_lookback = score_lookback
        self.val_dataset = val_dataset
        self.val_metric = val_metric
        self.verbose = verbose
        self.epoch = 0
        self.losses = []
        self.score = None

    def evaluate(self, pipeline: 'Pipeline') -> float:
        prediction = pipeline.predict(self.val_dataset.X)
        target = self.val_dataset.y.cpu().numpy()
        score = self.val_metric(target, prediction)
        return score

    def update(self, pipeline: 'Pipeline'):
        """
        Callback called by Pipeline every epoch during training
        """
        self.score = self.evaluate(pipeline)
        self.losses.append(self.score)
        self.epoch += 1
        if self.epoch % self.score_lookback == 0 and self.verbose:
            average_score = np.array(self.losses)[-self.score_lookback:].mean()
            print(f'Epoch {self.epoch:d}: {average_score:.4f}')

    def reset(self):
        self.epoch = 0
        self.losses = []
        self.score = None

    def plot_losses(self):
        plt.plot(self.losses, 'o-')
        plt.xlabel('Validation Loss')
        plt.ylabel('Epoch')

class Pipeline:
    """
    Analogue of sklearn or skorch pipeline that implements fit and predict.
    Attributes:
        model: underlying nn.Module neural network
        loss_fn: loss is computed as loss_fn(yhat.squeeze(), y), where yhat = model(X)
        activation: function applied to raw NN output before returning in predict method e.g. sigmoid
        optimizer_params: unpacked and passed to AdamW init as keyword arguments
        batch_size: training batch size
        epochs: training epochs
        seed: seed for data shuffle only; global seeds should be set prior to training for reproducible results
        training_device: torch.device for training; data will be moved here (if necessary) in batches
        scheduler_name: name of class in torch.optim.lr_scheduler to use as scheduler; called every batch
        scheduler_params: unpacked and passed to scheduler init as keyword arguments
    """
    def __init__(
            self,
            model: nn.Module,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], float],
            activation: nn.Module,
            optimizer_params: Dict,
            batch_size: int,
            epochs: int,
            seed: int,
            training_device: torch.device,
            scheduler_name: Optional[str] = None,
            scheduler_params: Optional[Dict] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.activation = activation
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.optimizer = None
        self.scheduler = None
        self.training_device = training_device
        self._predict_batch_size = 1024
    
    @staticmethod
    def _batch_iterator(X: torch.Tensor, batch_size: int, drop_last: Optional[bool] = True) -> Iterator[torch.Tensor]:
        """
        Used in training and predict method to loop over batches
        """
        if drop_last:
            N_batches = int(np.floor(len(X)/batch_size))
        else:
            N_batches = int(np.ceil(len(X)/batch_size))
        b = 0
        while b < N_batches:
            yield X[b*batch_size:(b+1)*batch_size]
            b += 1
    
    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Uses the model to predict on a torch.Tensor, returning a (N, ) ndarray (where N is the number of samples)
        """
        self.model.eval()
        prediction = []
        for batch in self._batch_iterator(X,  self._predict_batch_size, drop_last=False):
            batch = batch.to(self.training_device)
            prediction.append(self.model(batch).squeeze())
        prediction = self.activation(torch.cat(prediction)).cpu().numpy()
        return prediction
    
    def _init_training(self, dataset: TabularDataset) -> None:
        """
        Called before the start of training
        """
        self._init_optimizer()
    
    def _init_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.optimizer_params
            )
        if self.scheduler_name is not None:
            self.scheduler = vars(lr_scheduler)[self.scheduler_name](
                self.optimizer,
                **self.scheduler_params,
                )

    def _loss_from_batch(
            self,
            batch: Union[List[torch.Tensor], Tuple[torch.Tensor]]
    ) -> torch.Tensor:
        x_batch = batch[0].to(self.training_device)
        y_batch = batch[1].to(self.training_device)
        yhat = self.model(x_batch)
        return self.loss_fn(yhat.squeeze(), y_batch)
    
    def train(self, dataset: TabularDataset, tracker: Optional[ProgressTracker] = None) -> None:
        rng = np.random.default_rng(self.seed)
        self._init_training(dataset)
        for epoch in range(self.epochs):
            idxs = rng.choice(len(dataset), size=len(dataset), replace=False)
            for batch_idx in self._batch_iterator(idxs, self.batch_size, drop_last=True):
                batch = dataset[batch_idx]
                self.model.train()
                loss = self._loss_from_batch(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            if tracker is not None:
                tracker.update(self)

class FeatureEmbeddingPipeline(Pipeline):
    """
    Pipeline that makes the model (nntd.modules.FeatureEmbeddingModel), loss function, and activation function from a
    dict of parameters instead of requiring them to be constructed separately.
    Attributes:
        See Pipeline attributes except for:
        model_params: Dict of model keyword arguments used to make the model (see _init_training)
    """
    def __init__(
            self,
            model_params: Dict,
            optimizer_params: Dict,
            task_type: str,
            batch_size: int,
            epochs: int,
            seed: int,
            training_device: torch.device,
            scheduler_name: Optional[str] = None,
            scheduler_params: Optional[Dict] = None,
            augmentation_params: Optional[Dict] = None,
    ) -> None:
        """
        The task_type argument is used to automatically construct the relevant loss_fn and activation which are passed
        to super().__init__(). model = None is passed to super().__init__(), it will be created at train-time by
        _init_training.
        """
        if task_type == 'binclass':
            loss_fn = nn.BCEWithLogitsLoss()
            activation = nn.Sigmoid()
        elif task_type == 'regression':
            loss_fn = nn.MSELoss()
            activation = nn.Identity() 
        super().__init__(
            None,
            loss_fn,
            activation, 
            optimizer_params,
            batch_size,
            epochs,
            seed,
            training_device,
            scheduler_name,
            scheduler_params,
            )
        self.model_params = model_params
        self.augmentation_params = augmentation_params
    
    def _init_training(self, dataset: TabularDataset) -> None:
        """
        Construct optimizer, scheduler, and model. See modules.FeatureEmbeddingModel for keys of model_params.
        The (training) dataset is passed here since it is required to calculate the bin edges of binning encodings.
        """
        self.model = FeatureEmbeddingModel.make(dataset, **self.model_params, d_out=1)
        if self.augmentation_params is not None:
            self.model = AugmentedFeatureEmbeddingModel(self.model, **self.augmentation_params)
        self.model.to(self.training_device)
        self._init_optimizer()
