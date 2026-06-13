from __future__ import annotations

from typing import cast, override

import torch
from torch import nn


class PlainGRUPoseClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 192,
        sequence_length: int = 192,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 7,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if num_layers == 1:
            dropout = 0.0
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.bidirectional = bidirectional
        feature_size = hidden_size * (2 if bidirectional else 1)
        self.gru: nn.GRU = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.classifier: nn.Linear = nn.Linear(feature_size, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_shape = (self.sequence_length, self.input_size)
        if x.ndim != 3 or x.shape[1:] != expected_shape:
            raise ValueError(f"expected input shape (B, {self.sequence_length}, {self.input_size}), got {tuple(x.shape)}")
        _output, hidden = self.gru(x)
        hidden_tensor = cast(torch.Tensor, hidden)
        if self.bidirectional:
            final_hidden = torch.cat((hidden_tensor[-2], hidden_tensor[-1]), dim=1)
        else:
            final_hidden = hidden_tensor[-1]
        logits = self.classifier(final_hidden)
        return cast(torch.Tensor, logits)
