from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None
    nn = None


if nn is not None:

    class PositionalEncoding(nn.Module):
        def __init__(self, model_dim: int, max_len: int = 512) -> None:
            super().__init__()
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
            encoding = torch.zeros(max_len, model_dim)
            encoding[:, 0::2] = torch.sin(position * div_term)
            encoding[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.encoding[:, : x.size(1)]


    class DemandForecastTransformer(nn.Module):
        def __init__(
            self,
            *,
            input_dim: int,
            model_dim: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_projection = nn.Linear(input_dim, model_dim)
            self.positional_encoding = PositionalEncoding(model_dim=model_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_head = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim // 2),
                nn.ReLU(),
                nn.Linear(model_dim // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.input_projection(x)
            encoded = self.positional_encoding(encoded)
            encoded = self.encoder(encoded)
            context = encoded[:, -1, :]
            return self.output_head(context).squeeze(-1)

else:

    class PositionalEncoding:  # pragma: no cover - environment dependent
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to instantiate the transformer forecasting model.")


    class DemandForecastTransformer:  # pragma: no cover - environment dependent
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to instantiate the transformer forecasting model.")
