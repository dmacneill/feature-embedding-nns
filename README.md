### Feature Embedding Neural Networks for Tabular Data
PyTorch implementation of neural network architectures from ["On Embeddings for Numerical Features in Tabular Deep Learning"](https://arxiv.org/abs/2203.05556). The backbone modules are imported from the [rtdl](https://github.com/Yura52/rtdl) package. The feature embedding modules are re-implemented here. Compared to the paper there are several major differences:
1. Native support for NaNs in numeric features: they are mapped by the embedding layers to a feature-dependent vector
2. Embeddings for categorical variables as in ["Entity Embeddings of Categorical Variables"](https://arxiv.org/abs/1604.06737)
3. Masking: ability to replace feature embeddings by a mask vector
### Requirements
Python 3, PyTorch, NumPy, rtdl
### Usage
The models can be trained with a few lines of code:
```
tracker = nntd.ProgressTracker(score_lookback, val_dataset, val_metric)
pipeline = nntd.FeatureEmbeddingPipeline(**hyperparameter_dict, training_device = training_device)
pipeline.train(train_dataset, tracker)
```
However, setting up the hyperparameter dict and datasets is a bit complicated. See `examples/nntd_example.ipynb` or `examples/hyperparameters` for examples.
### Example Results

I wanted to test the performance of the built-in NaN handling. I trained and evaluted several models on the [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) dataset. For every experiment, hyperparameters were chosen using the validation set and final performance was evaluated on the test set. This is just a quick test. The results will likely change with further optimization and more realistic NaNs/datasets.

As a baseline I trained XGBoost and a "ResNet-QLR" neural network on the clean dataset ("No NaNs" below). For the other experiments, I randomly replaced 10% of feature values (excluding Latitude/Longitude) with NaN. The NaN-handling methods were:
1. Median impute
2. Built-in method: default directions for XGBoost and NaN embeddings for the neural network
3. Augmentation: same as 2, but 15% of feature values are randomly replaced with NaNs in each batch (NN only)

The results are summarized below. The averages and ensembles are created by re-training the models with 5 different seeds.

|   | No NaNs | Median Impute | Built-in | Augmentation |
|:---:|:---:|:---:|:---:|:---:|
| XGBoost Average RMSE | 0.445 | 0.461 | 0.460 | NA |
| ResNet-QLR Average RMSE | 0.450 | 0.476 | 0.479 | 0.455 |
| XGBoost Ensemble RMSE | 0.444 | 0.460 | 0.458 | NA |
| ResNet-QLR  Ensemble RMSE | 0.443 | 0.467 | 0.470 | 0.447 |

The NaN augmentation significantly improves the NN performance on data with NaNs; this is intuitive because it will help to train the NaN embeddings. One problem with this augmentation is that it could bias the network towards treating NaNs as MCAR, which is not a good assumption in general.
