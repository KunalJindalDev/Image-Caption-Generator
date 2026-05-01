from __future__ import annotations

import torch
from torch import Tensor, nn


class SoftAttention(nn.Module):
    """Soft additive attention over spatial encoder features."""

    def __init__(self, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        attention_dim = 512
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim)
        self.score_projection = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out: Tensor, decoder_hidden: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            encoder_out: Shape (B, 49, encoder_dim).
            decoder_hidden: Shape (B, decoder_dim).

        Returns:
            context: Shape (B, encoder_dim).
            alpha: Shape (B, 49).
        """
        encoder_proj = self.encoder_projection(encoder_out)  # (B, 49, 512)
        decoder_proj = self.decoder_projection(
            decoder_hidden).unsqueeze(1)  # (B, 1, 512)

        scores = self.score_projection(torch.tanh(
            encoder_proj + decoder_proj)).squeeze(2)  # (B, 49)
        alpha = torch.softmax(scores, dim=1)  # (B, 49)

        # (B, encoder_dim)
        context = (alpha.unsqueeze(2) * encoder_out).sum(dim=1)
        return context, alpha
