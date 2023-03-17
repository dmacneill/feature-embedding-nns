import numpy as np
import torch
import torch.nn as nn
import rtdl
from rtdl import MLP, ResNet
from .dataset import TabularDataset
from typing import List, Union, Dict, Optional, Tuple

def init_parameter_uniform(parameter: nn.Parameter, n: int) -> None:
    nn.init.uniform_(parameter, -1/np.sqrt(n), 1/np.sqrt(n))

class NLinear(nn.Module):
    """
    Module adapted from rtdl.nn._embeddings. This does a "feature-wise" linear map. If we have an input with shape
    (n_batch, n_features, d_in) this module applies a different linear map to each feature vector creating a
    (n_batch, n_features, d_out) result.
    Attributes:
        weight: (n_features, d_in, d_out) weights of linear maps
        bias: (n_features, d_out) biases of linear maps
    """
    def __init__(self, n_tokens: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_tokens, d_in, d_out))
        self.bias = nn.Parameter(torch.Tensor(n_tokens, d_out))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        d_out = self.weight.shape[-1]
        init_parameter_uniform(self.weight, d_out)
        init_parameter_uniform(self.bias, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features, d_in)
        returns: (n_batch, n_features, d_out)
        """
        x = (x.unsqueeze(-2)@self.weight[None]).squeeze(-2)
        x = x + self.bias[None]
        return x

class PiecewiseLinearEncoder(nn.Module):
    """
    Create piecewise linear encoding of numerical features as described in
    "On Embeddings for Numerical Features in Tabular Deep Learning". Unlike the rtdl version, we concatenate a 0/1
    non-null/null indicator with the encoding. If x=nan the encoding is (1, 0, 0, ...).
    Attributes:
        d_encoding: the size of the encoding vector for each feature
    """
    @staticmethod
    def compute_quantile_bin_edges(
            data: torch.Tensor,
            n_bins: Union[List[int], int],
            device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Calculates a list of bin_edges for each feature that bins the data into quantiles. Any duplicate bin_edges are
        dropped.
        Args:
            data: torch.Tensor of numerical data
            n_bins: number of bins, can be a list of values for each column of data
            device: torch.device that the data is on
        Return:
            List of bin_edges computed to bin each feature into quantiles
        """
        if isinstance(n_bins, int):
            n_bins = [n_bins]*data.shape[1]
        edges = []
        for i, n_bins_column in enumerate(n_bins):
            quantiles = torch.linspace(0.0, 1.0, n_bins_column + 1, device=device)
            edges_column = torch.unique(torch.nanquantile(data[:, i], quantiles))
            edges.append(edges_column)
        return [x.cpu() for x in edges]
    
    @classmethod
    def make(cls, dataset: TabularDataset, d_encoding: int) -> nn.Module:
        """
        Alternate constructor called by FeatureEmbedding model.
        """
        bin_edges = cls.compute_quantile_bin_edges(
            dataset.get_numeric_features(),
            d_encoding,
            dataset.storage_device,
            )
        return cls(bin_edges)
    
    def __init__(self, bin_edges: List[torch.Tensor]) -> None:
        super().__init__()
        self.d_encoding = max(map(len, bin_edges))#n_bins+1, which is the correct value including the NaN indicator
        bin_edge_tensor = self._make_bin_edges_tensor(bin_edges)
        lower_bounds = bin_edge_tensor[0, :, :-1]
        upper_bounds = bin_edge_tensor[0, :, 1:]
        slopes = torch.nan_to_num((1/(upper_bounds-lower_bounds)))
        self.register_buffer('_lower_bounds', lower_bounds)
        self.register_buffer('_upper_bounds', upper_bounds)
        self.register_buffer('_slopes', slopes)
        self.relu = nn.ReLU()
    
    def _make_bin_edges_tensor(self, bin_edges: List[torch.Tensor]) -> torch.Tensor:
        """
        Converts the list of bin_edge tensors into a single tensor. Since the bin_edges for different features could
        have different lengths, they are padded to be the same length with np.inf.
        """
        bin_edge_tensor = torch.ones((1, len(bin_edges), self.d_encoding))*np.inf
        for i, edges in enumerate(bin_edges):
            bin_edge_tensor[0, i, :len(edges)] = edges
        return bin_edge_tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features)
        returns: (n_batch, n_features, d_encoding)
        """
        nan_mask = x.isnan().float()
        x = x[..., None]
        x = self.relu(x-self._lower_bounds)-self.relu(x-self._upper_bounds)
        x = torch.nan_to_num(self._slopes*x)
        return torch.cat((nan_mask[..., None], x), dim=-1)

class PeriodicEncoder(nn.Module):
    """
    Encodes numeric feature by creating a list of fourier features (cos(w_0*x), cos(w_1*x), ..., sin(w_0*x), ...) as
    described in "On Embeddings for Numerical Features in Tabular Deep Learning". Each feature has a unique set of
    frequencies. Unlike the rtdl version, we concatenate a 0/1 non-null/null indicator with the encoding. If x=nan the
    encoding is (1, 0, 0, ...).
    Attributes:
        d_encoding: the size of the encoding vector for each feature; always odd
        sigma: standard deviation of frequencies on initialization
        coefficients: (n-features, d_encoding//2)
    """
    @classmethod
    def make(cls, dataset: TabularDataset, d_encoding: int, sigma: float) -> nn.Module:
        """
        Alternate constructor called by FeatureEmbedding model.
        dataset: not used, included for compatability
        """
        return cls(dataset.n_num_features, d_encoding, sigma)
    
    def __init__(self, n_num_features: int, d_encoding: int, sigma: float) -> None:
        """
        Args:
            n_num_features: number of numeric features
            d_encoding: number of cosine+sine features in the encoding; rounded down to an even number
            sigma: standard deviation of frequencies on initialization
        """
        super().__init__()
        self.d_encoding = 2*(d_encoding//2)+1#Add one dimension to encode NaNs
        self.sigma = sigma
        coefficients = torch.Tensor(n_num_features, d_encoding//2)
        self.coefficients = nn.Parameter(coefficients)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.normal_(self.coefficients, 0.0, self.sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features)
        returns: (n_batch, n_features, d_encoding)
        """
        nan_mask = x.isnan().float()
        x = torch.nan_to_num(2*np.pi*self.coefficients[None]*x[..., None])
        return torch.cat([nan_mask[..., None], torch.cos(x), torch.sin(x)], -1)

class PLU(nn.Module):
    """
    Modified piecewise linear unit (saturating ReLU)
    Attributes:
        c: x value where the linear response saturates
        alpha: slope after the saturation point
    The output is alpha*x for x<0, x for 0<x<c, and alpha*(x-c)+c for x>c.
    """
    def __init__(self, c: float, alpha: float) -> None:
        super().__init__()
        self.c = c
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_minus = self.alpha*x
        x_plus = self.alpha*(x-self.c)+self.c
        return torch.max(x_minus, torch.min(x_plus, x))

class PLUEncoder(nn.Module):
    """
    Creates numerical encoding by applying a feature dependent linear map and PLU activation. Similar to "LR" (linear
    plus ReLU) encoding from "On Embeddings for Numerical Features in Tabular Deep Learning".
    Attributes:
        d_encoding: the size of the encoding vector for each feature; always odd
        sigma: bounds on initialization of weights and biases
        c: PLU c parameter
        alpha: PLU alpha parameter
        weight: (n_features, d_encoding) weight tensor
        bias: (n_features, d_encoding) biases tensor
        activation: PLU activation
    """
    @classmethod
    def make(
            cls,
            dataset: TabularDataset,
            d_encoding: int,
            sigma: float,
            c: float = 1,
            alpha: float = 1e-2
    ) -> nn.Module:
        """
        Alternate constructor called by FeatureEmbedding model.
        dataset: not used, included for compatability
        """
        return cls(dataset.n_num_features, d_encoding, sigma, c, alpha)

    def __init__(self, n_num_features: int, d_encoding: int, sigma: float, c: float, alpha: float) -> None:
        super().__init__()
        self.d_encoding = d_encoding
        self.c = c
        self.alpha = alpha
        self.sigma = sigma
        self.bias = nn.Parameter(torch.Tensor(n_num_features, d_encoding))
        self.weight = nn.Parameter(torch.Tensor(n_num_features, d_encoding))
        self.activation = PLU(c, alpha)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.bias, -self.sigma, self.sigma)
        nn.init.uniform_(self.weight, -self.sigma, self.sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features)
        returns: (n_batch, n_features, d_encoding)
        """
        x = self.weight[None] * x[..., None] + self.bias[None]
        return self.activation(x)

class IdentityEncoder(nn.Module):
    """
    Placeholder encoder that returns input with new dimension of size 1 added. Used to make model without an encoder.
    """
    @classmethod
    def make(cls, dataset: TabularDataset) -> nn.Module:
        """
        Alternate constructor called by FeatureEmbedding model.
        dataset: not used, included for compatability
        """
        return cls()

    def __init__(self) -> None:
        super().__init__()
        self.d_encoding = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features)
        returns: (n_batch, n_features, 1)
        """
        return x.unsqueeze(-1)

class MaskEmbedding(nn.Module):
    """
    Replaces feature embedding with a mask embedding.
    """
    def __init__(self, d_token: int,  n_tokens: int) -> None:
        """
        d_token: dimension of the mask embedding
        n_tokens: set 1 to use a shared mask embedding for all features, n_features to use feature-dependent ones
        """
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_tokens, d_token))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init_parameter_uniform(self.weight, self.weight.shape[1])
    
    def forward(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        mask: (n_batch, n_features)
        x: (n_batch, n_features, d_token)
        returns: (n_batch, n_features, d_token)
        if mask[i][j]==True, then x[i][j][:] will be replaced by the mask embedding
        """
        if mask is None:
            return x
        return torch.where(mask[..., None], self.weight[None], x)

class NumericalEmbedding(nn.Module):
    """
    Creates the numerical embedding. Takes (n_batch, n_features) tensor of numeric data and transforms it
    to a (n_batch, n_features, d_num) or (n_batch, n_features*d_num) tensor of embeddings.
    Attributes:
        encoder: module that maps (n_batch, n_features)->(n_batch, n_features, d_encoding)
        n_num_features: number of features expected
        flat: if True, output the embedding as (n_batch, n_features*d_num)
        d_num: size of embedding
    """
    @classmethod
    def make(cls, encoder: nn.Module, n_num_features: int, flat: bool, d_num: int) -> nn.Module:
        """
        Alternate constructor called by FeatureEmbedding model.
        """
        return cls(encoder, n_num_features, flat, d_num)
    
    def __init__(self, encoder: nn.Module, n_num_features: int, flat: bool, d_num: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_num_features = n_num_features
        self.d_num = d_num
        self.nlinear = NLinear(
            n_tokens=n_num_features,
            d_in=self.encoder.d_encoding,
            d_out=d_num,
            )
        self.activation = nn.ReLU()
        self.mask_embedding = MaskEmbedding(self.d_num, 1)
        self.flat = flat
    
    def _raw_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.nlinear(x)
        x = self.activation(x)
        return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (n_batch, n_features)
        mask: (n_batch, n_features)
        returns: (n_batch, n_features, d_num) or (n_batch, n_features*d_num)
        """
        x = self._raw_embedding(x)
        x = self.mask_embedding(mask, x)
        x = x.reshape(len(x), -1) if self.flat else x
        return x

class CategoricalEmbedding(nn.Module):
    """
    Creates the categorical embedding. Takes (n_batch, n_features) tensor of categorical data and transforms it
    to a (n_batch, n_features, d_cat) or (n_batch, sum(d_i)) tensor of embeddings. In the flattened output case,
    it is possible to have embeddings of different size (d_i) for different categorical features.
    The inputs are assumed to be encoded as integers 0 to n_classes-1 for each categorical feature.
    Attributes:
        embeddings: all the embeddings stored as a (sum(cardinalities), d_cat) Tensor
        flat: flat or stacked output
        embedding_dims: list of embedding sizes for each feature; only used if flat=True. If the output is stacked,
        then all the embeddings have the maximum size d_cat.
    """
    @classmethod
    def make(cls, cardinalities: List[int], flat: bool, d_cat: int, bias: Optional[bool] = True) -> nn.Module:
        """
        Alternate constructor called by FeatureEmbedding model
        cardinalities: list of number of classes for each categorical feature
        flat: flatten the output to (n_batch, sum(d_i*card_i)) if True
        d_cat: maximum categorical embedding size if flat=True; cat embedding size if flat=False
        bias: if True, a trainable offset vector is added to all embeddings for a feature
        """
        embedding_dims = [min(d_cat, card) for card in cardinalities]
        return cls(cardinalities, flat, d_cat, embedding_dims, bias)
    
    def __init__(
            self,
            cardinalities: List[int],
            flat: bool,
            d_cat: int,
            embedding_dims: List[int],
            bias: bool,
    ) -> None:
        super().__init__()
        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets)
        self.embeddings = nn.Embedding(sum(cardinalities), d_cat)
        self.bias = nn.Parameter(torch.Tensor(len(cardinalities), d_cat)) if bias else None
        self.mask_embedding = MaskEmbedding(d_cat, 1)
        self.flat = flat
        take_idxs = self._calculate_take_idxs(cardinalities, d_cat)
        self.register_buffer('_take_idxs', take_idxs)
        self.embedding_dims = embedding_dims
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        d_cat = self.embeddings.weight.shape[1]
        init_parameter_uniform(self.embeddings.weight, d_cat)
        if self.bias is not None:
            init_parameter_uniform(self.bias, d_cat)

    @staticmethod
    def _calculate_take_idxs(embedding_dims: List[int], d_cat: int) -> torch.LongTensor:
        take_idxs = []
        for i, d in enumerate(embedding_dims):
            take_idxs.append(torch.arange(d_cat*i, d_cat*i+min(d, d_cat)))
        return torch.cat(take_idxs).type(torch.long)
        
    def _raw_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.round(x + self.category_offsets[None]).type(torch.long)
        x = self.embeddings(x)
        if self.bias is not None:
            x = x + self.bias[None]
        return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (n_batch, n_features); feature values are assumed to be integer encodings of the categorical classes
        mask: (n_batch, n_features)
        returns: (n_batch, n_features, d_cat) or (n_batch, sum(d_i))
        """
        x = self._raw_embedding(x)
        x = self.mask_embedding(mask, x)
        if self.flat:
            x = x.reshape(len(x), -1)
            x = torch.take_along_dim(x, self._take_idxs[None], dim=1)
        return x

class FlatCategoricalEmbedding(nn.Module):
    """
    Unused/obsolete module. Same logical output as CategoricalEmbedding but only works to create flat embeddings.
    """
    @classmethod
    def make(cls, cardinalities: List[int], d_cat: int) -> nn.Module:
        embedding_dims = [min(d_cat, card) for card in cardinalities]
        return cls(cardinalities, embedding_dims)

    def __init__(self, cardinalities: List[int], embedding_dims: List[int]) -> None:
        super().__init__()
        base_idxs = self._get_base_idxs(cardinalities, embedding_dims)
        self.register_buffer('base_idxs', base_idxs)
        class_offset_idxs = self._get_offset_idxs(embedding_dims)
        self.register_buffer('class_offset_idxs', class_offset_idxs)
        embedding_parameter_count = sum(c*d for c, d in zip(cardinalities, embedding_dims))
        self.embeddings = nn.Parameter(torch.Tensor(1, embedding_parameter_count))
        self.embedding_dims = embedding_dims
        self.reset_parameters()

    def reset_parameters(self) -> None:
        n = sum(self.embedding_dims)/len(self.embedding_dims)
        init_parameter_uniform(self.embeddings, n)

    @staticmethod
    def _get_base_idxs(cardinalities: List[int], embedding_dims: List[int]) -> torch.Tensor:
        """
        Args:
            cardinalities: list of number of classes for each categorical feature
            embedding_dims: list of embedding sizes for each feature
        Returns:
            (1, sum(embedding_dims)) tensor that looks like [0, ..., d_0-1, n_0*d_0, n_0*d_0+1, ...], where d_i is the
            embedding dimension of categorical feature i, and n_i is the number of classes.
        """
        base_idxs = []
        start = 0
        for c, d in zip(cardinalities, embedding_dims):
            base_idxs.append(torch.arange(start, start+d, dtype=torch.float))
            start += c*d
        base_idxs = torch.cat(base_idxs)[None]
        return base_idxs
    @staticmethod
    def _get_offset_idxs(embedding_dims: List[int]) -> torch.Tensor:
        """
        Args:
            embedding_dims: list of embedding sizes for each feature
        Returns:
            (len(embedding_dims), sum(embedding_dims)) torch.Tensor that looks like diag(embedding_dims) with column i
            repeated d_i times.
        """
        row_idxs = np.concatenate([[i]*d for i, d in enumerate(embedding_dims)])
        class_offset_idxs = np.diag([d for d in embedding_dims])[row_idxs]
        class_offset_idxs = torch.tensor(class_offset_idxs, dtype=torch.float).T
        return class_offset_idxs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.round(self.base_idxs + x@self.class_offset_idxs).type(torch.long)
        return torch.take(self.embeddings, x)

class Transformer(rtdl.Transformer):
    """
    Wrapper of rtdl.Transformer implementing forward and make_baseline methods in a convenient way. See
    github.com/Yura52/rtdl or "Revisiting Deep Learning Models for Tabular Data".
    """
    @classmethod
    def make_baseline(
            cls,
            d_in: int,
            n_blocks: int,
            attention_dropout: float,
            ffn_d_hidden: int,
            ffn_dropout: float,
            d_out: int,
    ) -> nn.Module:
        """
        Alternate constructor called by FeatureEmbedding model
        d_in: embedding dimension (d_cat==d_num for non-flat model)
        n_blocks: number of Transformer layers
        attention_dropout: dropout applied to attention probabilities
        ffn_d_hidden: size of feed-forward network hidden layer
        ffn_dropout: dropout applied to feed-forward network hidden activations
        d_out: dimension of head output
        """
        return cls(d_in, n_blocks, attention_dropout, ffn_d_hidden, ffn_dropout, d_out)

    def __init__(
            self,
            d_in: int,
            n_blocks: int,
            attention_dropout: float,
            ffn_d_hidden: int,
            ffn_dropout: float,
            d_out: int,
    ) -> None:
        super().__init__(
            d_token=d_in,
            n_blocks=n_blocks,
            attention_n_heads=8,
            attention_dropout=attention_dropout,
            attention_initialization='kaiming',
            attention_normalization='LayerNorm',
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=ffn_dropout,
            ffn_activation='ReGLU',
            ffn_normalization='LayerNorm',
            residual_dropout=0.0,
            prenormalization=True,
            first_prenormalization=False,
            last_layer_query_idx=[-1],
            n_tokens=None,
            kv_compression_ratio=None,
            kv_compression_sharing=None,
            head_activation='ReLU',
            head_normalization='LayerNorm',
            d_out=d_out,
        )
        self.cls_token = rtdl.CLSToken(d_in, 'uniform')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features, d_in)
        returns: (n_batch, d_out)
        """
        x = self.cls_token(x)
        return super().forward(x)

class FeatureEmbeddingModel(nn.Module):
    """
    Neural network model for tabular data based on "On Embeddings for Numerical Features in Tabular Deep Learning".
    Has three main submodules: num_tokenizer, cat_tokenizer, and backbone. num_tokenizer/cat_tokenizer are modules
    applied to each feature independently converting (n_batch, n_num_features+n_cat_features) input data
    into a (n_batch, n_num_features+n_cat_features, d_embedding) or (n_batch, n_num_features*d_num+sum(d_cat_i))
    embedding. The backbone modules are taken from rtdl: ResNet, MLP and Transformer are supported.
    """
    @classmethod
    def make(
            cls,
            dataset: TabularDataset,
            encoder_name: str,
            encoder_params: Dict,
            d_num: int,
            d_cat: Optional[int],
            dropout_input: float,
            backbone_name: str,
            backbone_params: Dict,
            d_out: int,
            flat_embedding: Optional[bool] = True,
            cat_bias: Optional[bool] = True,
    ):
        """
        Called by FeatureEmbeddingPipeline to initialize the model
        dataset: training dataset; used to construct some of the encoders and has necessary metadata
        encoder_name: name of class to use for encoder (e.g PiecewiseLinearEncoder or PeriodicEncoder)
        encoder_params: dict of keyword arguments to encoder.make
        d_num: size of numeric embedding
        d_cat: maximum size of categorical embedding; must be = d_num for stacked models
        dropout_input: dropout applied to input to backbone
        backbone_name: name of class to use for backbone
        backbone_params: dict of keyword arguments to backbone.make_baseline
        d_out: output dimension of backbone
        flat_embedding: True if the embeddings should be flattened (e.g. ResNet or MLP backbone)
        cat_bias: True to use a bias for the categorical embeddings
        """
        assert(
            not (dataset.cardinalities and d_cat is None)
        ), 'd_cat must be set if there are categorical features'
        assert(
            not (flat_embedding and backbone_name == 'Transformer')
        ), 'flat_embedding must be False for the Transformer backbone'
        encoder = globals()[encoder_name].make(dataset=dataset, **encoder_params)
        num_tokenizer = NumericalEmbedding.make(
            encoder,
            dataset.n_num_features,
            flat=flat_embedding,
            d_num=d_num,
            )
        d_in = num_tokenizer.d_num*dataset.n_num_features
        if dataset.cardinalities:
            cat_tokenizer = CategoricalEmbedding.make(
                dataset.cardinalities,
                flat=flat_embedding,
                d_cat=d_cat,
                bias=cat_bias,
                )
            d_in += sum(cat_tokenizer.embedding_dims)
        else:
            cat_tokenizer = None
        d_in = d_in if flat_embedding else num_tokenizer.d_num
        backbone = globals()[backbone_name].\
            make_baseline(**{**{'d_in': d_in, 'd_out': d_out}, **backbone_params})
        return cls(
            num_tokenizer,
            cat_tokenizer,
            dropout_input,
            backbone,
            )
    
    def __init__(
        self,
        num_tokenizer: nn.Module,
        cat_tokenizer: nn.Module,
        dropout_input: float,
        backbone: nn.Module,
    ) -> None:
        super().__init__()
        self.n_num_features = num_tokenizer.n_num_features
        self.num_tokenizer = num_tokenizer
        self.cat_tokenizer = cat_tokenizer
        self.dropout_input = nn.Dropout(dropout_input)
        self.backbone = backbone

    def split_features(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split feature tensor into numeric and categorical parts
        """
        x_num, x_cat = x[:, :self.n_num_features], x[:, self.n_num_features:]
        if mask is not None:
            num_mask, cat_mask = mask[:, :self.n_num_features], mask[:, self.n_num_features:]
        else:
            num_mask, cat_mask = None, None
        return x_num, x_cat, num_mask, cat_mask

    def forward_from_split_features(
            self,
            x_num: torch.Tensor,
            x_cat: torch.Tensor,
            num_mask: Optional[torch.Tensor] = None,
            cat_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.num_tokenizer(x_num, num_mask)
        if self.cat_tokenizer is not None:
            x_cat = self.cat_tokenizer(x_cat, cat_mask)
            x = torch.cat((x, x_cat), dim=1)
        x = self.dropout_input(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (n_batch, n_num_features+n_cat_features); the first n_num_features columns are assumed to be the numeric
        features. The categorical features are assumed to be integer encoded from 0 to n_classes-1
        mask: (n_batch, n_num_features+n_cat_features)
        returns: (n_batch, d_out)
        """
        x_num, x_cat, num_mask, cat_mask = self.split_features(x, mask)
        return self.forward_from_split_features(x_num, x_cat, num_mask, cat_mask)

class NaNAugmentation(nn.Module):
    """
    Module that adds NaNs to a batch of features
    Attributes:
        p_nan: probability of replacing numeric feature with NaN, or list of values for every numeric feature
        n_num_features: number of numeric features in the dataset
    """
    def __init__(self, p_nan: Union[List[float], float], n_num_features: int) -> None:
        if isinstance(p_nan, float):
            p_nan = [p_nan]*n_num_features
        assert(
            len(p_nan) == n_num_features
        ), 'len(p_nan) must equal the number of features if a list is passed'
        super().__init__()
        p_nan = torch.tensor([p_nan])
        self.register_buffer('p_nan', p_nan)
        self.n_num_features = n_num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            device = self.p_nan.device
            mask = torch.rand((len(x), self.n_num_features), device=device) < self.p_nan
            x = torch.where(mask, np.nan, x)
        return x

class GaussianNoise(nn.Module):
    """
    Module that adds Gaussian noise to numeric data
        sigma: noise standard deviation, or list of values for every numeric feature
        n_num_features: number of numeric features in the dataset
    """
    def __init__(self, sigma: Union[List[float], float], n_num_features: int) -> None:
        if isinstance(sigma, float):
            sigma = [sigma]*n_num_features
        assert(
            len(sigma) == n_num_features
        ), 'len(sigma) must equal the number of features if a list is passed'
        super().__init__()
        sigma = torch.tensor([sigma])
        self.register_buffer('sigma', sigma)
        self.n_num_features = n_num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            device = self.sigma.device
            noise = self.sigma*torch.normal(0, 1, (len(x), self.n_num_features), device=device)
            x = x+noise
        return x

class AugmentedFeatureEmbeddingModel(nn.Module):
    """
    Module that wraps FeatureEmbeddingModel and applies data augmentation to the input.
    Attributes:
        model: FeatureEmbeddingModel
        nan_augmentation: module that randomly adds NaNs to numeric data
        gaussian_noise: module that adds gaussian noise to numeric data
    """
    def __init__(
            self,
            model: FeatureEmbeddingModel,
            p_nan: Optional[Union[List[float], float]] = None,
            sigma: Optional[Union[List[float], float]] = None,
    ):
        """
        Args:
            model: FeatureEmbeddingModel
            p_nan: probability of replacing numeric feature with NaN, or list of values for every numeric feature
            sigma: noise standard deviation, or list of values for every numeric feature
        """
        super().__init__()
        self.nan_augmentation = None if p_nan is None else NaNAugmentation(p_nan, model.n_num_features)
        self.gaussian_noise = None if sigma is None else GaussianNoise(sigma, model.n_num_features)
        self.model = model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        See FeatureEmbeddingModel.forward
        """
        x_num, x_cat, num_mask, cat_mask = self.model.split_features(x, mask)
        if self.nan_augmentation is not None:
            x_num = self.nan_augmentation(x_num)
        if self.gaussian_noise is not None:
            x_num = self.gaussian_noise(x_num)
        return self.model.forward_from_split_features(x_num, x_cat, num_mask, cat_mask)
