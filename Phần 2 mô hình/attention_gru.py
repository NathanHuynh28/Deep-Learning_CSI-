from __future__ import annotations

from typing import cast, override

import torch
from torch import nn


class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int) -> None:
        super().__init__()
        self.score: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    @override
    def forward(self, gru_outputs: torch.Tensor) -> torch.Tensor:
        if gru_outputs.ndim != 3:
            raise ValueError(f"expected GRU outputs with shape (B, T, H), got {tuple(gru_outputs.shape)}")
        score_output = cast(torch.Tensor, self.score(gru_outputs))
        scores = score_output.squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(gru_outputs * weights.unsqueeze(-1), dim=1)
        return context


class AttentionGRUPoseClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 192,
        sequence_length: int = 192,
        hidden_size: int = 96,
        attention_dim: int = 32,
        num_layers: int = 1,
        num_classes: int = 7,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        recurrent_dropout = 0.0 if num_layers == 1 else dropout
        self.input_size: int = input_size
        self.sequence_length: int = sequence_length
        feature_size = hidden_size * (2 if bidirectional else 1)
        self.gru: nn.GRU = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        self.attention: TemporalAttentionPooling = TemporalAttentionPooling(hidden_size=feature_size, attention_dim=attention_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.classifier: nn.Linear = nn.Linear(feature_size, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_shape = (self.sequence_length, self.input_size)
        if x.ndim != 3 or x.shape[1:] != expected_shape:
            raise ValueError(f"expected input shape (B, {self.sequence_length}, {self.input_size}), got {tuple(x.shape)}")
        gru_result = self.gru(x)
        outputs = cast(torch.Tensor, gru_result[0])
        context = self.attention(outputs)
        logits = cast(torch.Tensor, self.classifier(self.dropout(context)))
        return cast(torch.Tensor, logits)
