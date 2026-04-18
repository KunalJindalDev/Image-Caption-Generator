from __future__ import annotations

import torch
from torch import Tensor, nn

from src.attention import SoftAttention


class DecoderWithAttention(nn.Module):
    """Attention-based caption decoder using an LSTMCell."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder_dim: int,
        decoder_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = SoftAttention(encoder_dim=encoder_dim, decoder_dim=decoder_dim)
        self.lstm_cell = nn.LSTMCell(input_size=embed_dim + encoder_dim, hidden_size=decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.output = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out: Tensor) -> tuple[Tensor, Tensor]:
        """Initialize hidden and cell states from mean encoder features."""
        mean_encoder = encoder_out.mean(dim=1)  # (B, encoder_dim)
        h = self.init_h(mean_encoder)  # (B, decoder_dim)
        c = self.init_c(mean_encoder)  # (B, decoder_dim)
        return h, c

    def forward(self, encoder_out: Tensor, captions: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            encoder_out: Encoder features of shape (B, 49, encoder_dim).
            captions: Token ids of shape (B, T).

        Returns:
            logits: Vocabulary logits of shape (B, T, vocab_size).
            alphas: Attention weights of shape (B, T, 49).
        """
        batch_size, timesteps = captions.size()

        embeddings = self.embedding(captions)  # (B, T, embed_dim)
        h, c = self.init_hidden_state(encoder_out)

        logits_steps: list[Tensor] = []
        alpha_steps: list[Tensor] = []

        for t in range(timesteps):
            emb_t = embeddings[:, t, :]  # (B, embed_dim)
            context, alpha = self.attention(encoder_out, h)  # (B, encoder_dim), (B, 49)
            lstm_input = torch.cat([emb_t, context], dim=1)  # (B, embed_dim + encoder_dim)
            h, c = self.lstm_cell(lstm_input, (h, c))

            step_logits = self.output(self.dropout(h))  # (B, vocab_size)
            logits_steps.append(step_logits)
            alpha_steps.append(alpha)

        logits = torch.stack(logits_steps, dim=1)  # (B, T, vocab_size)
        alphas = torch.stack(alpha_steps, dim=1)  # (B, T, 49)
        return logits, alphas

    @torch.no_grad()
    def generate(
        self,
        encoder_out: Tensor,
        max_len: int = 50,
        start_idx: int = 1,
        end_idx: int = 2,
    ) -> tuple[Tensor, Tensor]:
        """Greedy caption generation with per-sequence early stopping."""
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        h, c = self.init_hidden_state(encoder_out)
        current_tokens = torch.full((batch_size,), start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        generated_tokens: list[Tensor] = []
        attention_steps: list[Tensor] = []

        for _ in range(max_len):
            emb_t = self.embedding(current_tokens)  # (B, embed_dim)
            context, alpha = self.attention(encoder_out, h)  # (B, encoder_dim), (B, 49)
            lstm_input = torch.cat([emb_t, context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            step_logits = self.output(self.dropout(h))  # (B, vocab_size)
            next_tokens = step_logits.argmax(dim=1)  # (B,)

            # Keep ended sequences fixed at end token for stable batching.
            next_tokens = torch.where(finished, torch.full_like(next_tokens, end_idx), next_tokens)

            generated_tokens.append(next_tokens)
            attention_steps.append(alpha)

            finished = finished | (next_tokens == end_idx)
            if finished.all():
                break

            current_tokens = next_tokens

        if not generated_tokens:
            empty_tokens = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            empty_alpha = torch.empty((batch_size, 0, encoder_out.size(1)), dtype=encoder_out.dtype, device=device)
            return empty_tokens, empty_alpha

        tokens = torch.stack(generated_tokens, dim=1)  # (B, L)
        alphas = torch.stack(attention_steps, dim=1)  # (B, L, 49)
        return tokens, alphas
