from __future__ import annotations

from typing import cast, override

import torch
from torch import nn


class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int) -> None:
        super().__init__()
        self.score: nn.Sequential = nn.Sequential(nn.Linear(hidden_size, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))

    @override
    def forward(self, gru_outputs: torch.Tensor) -> torch.Tensor:
        if gru_outputs.ndim != 3:
            raise ValueError(f"expected GRU outputs with shape (B, T, H), got {tuple(gru_outputs.shape)}")
        scores = cast(torch.Tensor, self.score(gru_outputs)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(gru_outputs * weights.unsqueeze(-1), dim=1)


class MultiOutputAttentionGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 192,
        sequence_length: int = 192,
        hidden_size: int = 96,
        attention_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        cell_head_type: str = "linear",
        cell_head_hidden_size: int | None = None,
        cell_head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if cell_head_type not in {"linear", "mlp"}:
            raise ValueError("cell_head_type must be 'linear' or 'mlp'")
        if not 0.0 <= cell_head_dropout < 1.0:
            raise ValueError("cell_head_dropout must satisfy 0 <= cell_head_dropout < 1")
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
        self.presence_head: nn.Linear = nn.Linear(feature_size, 1)
        if cell_head_type == "mlp":
            cell_hidden = cell_head_hidden_size if cell_head_hidden_size is not None else max(feature_size // 2, 32)
            if cell_hidden < 1:
                raise ValueError("cell_head_hidden_size must be >= 1")
            self.cell_head: nn.Module = nn.Sequential(
                nn.LayerNorm(feature_size),
                nn.Dropout(cell_head_dropout),
                nn.Linear(feature_size, cell_hidden),
                nn.GELU(),
                nn.Dropout(cell_head_dropout),
                nn.Linear(cell_hidden, 25),
            )
        else:
            self.cell_head = nn.Linear(feature_size, 25)
        self.pose_head: nn.Linear = nn.Linear(feature_size, 7)
        self.center_head: nn.Linear = nn.Linear(feature_size, 2)

    @override
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_shape = (self.sequence_length, self.input_size)
        if x.ndim != 3 or x.shape[1:] != expected_shape:
            raise ValueError(f"expected input shape (B, {self.sequence_length}, {self.input_size}), got {tuple(x.shape)}")
        gru_result = self.gru(x)
        outputs = cast(torch.Tensor, gru_result[0])
        context = cast(torch.Tensor, self.dropout(self.attention(outputs)))
        return {
            "presence_logit": cast(torch.Tensor, self.presence_head(context).squeeze(-1)),
            "cell_logits": cast(torch.Tensor, self.cell_head(context)),
            "pose_logits": cast(torch.Tensor, self.pose_head(context)),
            "center_norm": torch.sigmoid(cast(torch.Tensor, self.center_head(context))),
        }
