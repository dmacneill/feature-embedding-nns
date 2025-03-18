from typing import TypeVar

import torch


class ResNetBlock(torch.nn.Module):
    def __init__(
            self,
            d_main: int,
            dropout_first: float,
            dropout_second: float,
            d_hidden: int
    ):
        super().__init__()
        self.normalization = torch.nn.BatchNorm1d(num_features=d_main)
        self.linear_1 = torch.nn.Linear(in_features=d_main, out_features=d_hidden)
        self.activation = torch.nn.ReLU()
        self.dropout_1 = torch.nn.Dropout(p=dropout_first)
        self.linear_2 = torch.nn.Linear(in_features=d_hidden, out_features=d_main)
        self.dropout_2 = torch.nn.Dropout(p=dropout_second)

    def forward(self, x: torch.tensor) -> torch.tensor:
        y = self.normalization(x)
        y = self.linear_1(y)
        y = self.activation(y)
        y = self.dropout_1(y)
        y = self.linear_2(y)
        y = self.dropout_2(y)
        y = x + y
        return y


class Head(torch.nn.Module):
    def __init__(
            self,
            d_main: int,
            d_out: int
    ):
        super().__init__()
        self.normalization = torch.nn.BatchNorm1d(num_features=d_main)
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(in_features=d_main, out_features=d_out)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear(x)
        return x


ResNetType = TypeVar('ResNetType', bound='ResNet')


class ResNet(torch.nn.Module):

    def __init__(
            self,
            d_in: int,
            d_out: int,
            d_main: int,
            n_blocks: int,
            dropout_first: float,
            dropout_second: float,
            d_hidden: int
    ):
        super().__init__()
        self.input_linear = torch.nn.Linear(d_in, d_main)
        self.blocks = torch.nn.Sequential(
            *[
                ResNetBlock(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = Head(
            d_main=d_main,
            d_out=d_out
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.input_linear(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

    @classmethod
    def make_baseline(
            cls,
            d_in: int,
            d_out: int,
            d_main: int,
            n_blocks: int,
            dropout_first: float,
            dropout_second: float,
            d_hidden: int
    ) -> ResNetType:
        model = cls(
            d_in=d_in,
            d_out=d_out,
            d_main=d_main,
            n_blocks=n_blocks,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            d_hidden=d_hidden
        )
        return model
