from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

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
        self.attention = SoftAttention(
            encoder_dim=encoder_dim, decoder_dim=decoder_dim)
        self.lstm_cell = nn.LSTMCell(
            input_size=embed_dim + encoder_dim, hidden_size=decoder_dim)

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
            context, alpha = self.attention(
                encoder_out, h)  # (B, encoder_dim), (B, 49)
            # (B, embed_dim + encoder_dim)
            lstm_input = torch.cat([emb_t, context], dim=1)
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
        current_tokens = torch.full(
            (batch_size,), start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        generated_tokens: list[Tensor] = []
        attention_steps: list[Tensor] = []

        for _ in range(max_len):
            emb_t = self.embedding(current_tokens)  # (B, embed_dim)
            context, alpha = self.attention(
                encoder_out, h)  # (B, encoder_dim), (B, 49)
            lstm_input = torch.cat([emb_t, context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            step_logits = self.output(self.dropout(h))  # (B, vocab_size)
            next_tokens = step_logits.argmax(dim=1)  # (B,)

            # Keep ended sequences fixed at end token for stable batching.
            next_tokens = torch.where(finished, torch.full_like(
                next_tokens, end_idx), next_tokens)

            generated_tokens.append(next_tokens)
            attention_steps.append(alpha)

            finished = finished | (next_tokens == end_idx)
            if finished.all():
                break

            current_tokens = next_tokens

        if not generated_tokens:
            empty_tokens = torch.empty(
                (batch_size, 0), dtype=torch.long, device=device)
            empty_alpha = torch.empty((batch_size, 0, encoder_out.size(
                1)), dtype=encoder_out.dtype, device=device)
            return empty_tokens, empty_alpha

        tokens = torch.stack(generated_tokens, dim=1)  # (B, L)
        alphas = torch.stack(attention_steps, dim=1)  # (B, L, 49)
        return tokens, alphas

    @torch.no_grad()
    def generate_beam(
        self,
        encoder_out: Tensor,
        beam_size: int = 3,
        max_len: int = 50,
        start_idx: int = 1,
        end_idx: int = 2,
    ) -> tuple[Tensor, Tensor]:
        """Beam-search caption generation.

        Args:
            encoder_out: Tensor of shape (B, 49, encoder_dim).
            beam_size: Number of candidate beams to keep.
            max_len: Maximum decoded length.
            start_idx: Start token id.
            end_idx: End token id.

        Returns:
            tokens: Tensor of shape (B, L_best).
            alphas: Tensor of shape (B, L_best, 49).
        """
        if beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {beam_size}")

        batch_tokens: list[Tensor] = []
        batch_alphas: list[Tensor] = []

        for b in range(encoder_out.size(0)):
            enc = encoder_out[b:b + 1]  # (1, 49, encoder_dim)
            h, c = self.init_hidden_state(enc)
            device = enc.device

            # (score, token_ids, alphas, h, c, finished)
            beams: list[tuple[float, list[int], list[Tensor], Tensor, Tensor, bool]] = [
                (0.0, [start_idx], [], h, c, False)
            ]
            completed: list[tuple[float, list[int], list[Tensor]]] = []

            for _ in range(max_len):
                candidates: list[tuple[float, list[int],
                                       list[Tensor], Tensor, Tensor, bool]] = []

                for score, seq, alpha_seq, h_i, c_i, finished in beams:
                    if finished:
                        candidates.append(
                            (score, seq, alpha_seq, h_i, c_i, True))
                        continue

                    prev_token = torch.tensor(
                        [seq[-1]], device=device, dtype=torch.long)
                    emb_t = self.embedding(prev_token)  # (1, embed_dim)
                    context, alpha = self.attention(
                        enc, h_i)  # (1, encoder_dim), (1, 49)
                    lstm_input = torch.cat([emb_t, context], dim=1)
                    h_next, c_next = self.lstm_cell(lstm_input, (h_i, c_i))

                    step_logits = self.output(
                        self.dropout(h_next))  # (1, vocab_size)
                    log_probs = F.log_softmax(step_logits, dim=1)

                    top_log_probs, top_indices = torch.topk(
                        log_probs, k=beam_size, dim=1)
                    for k in range(beam_size):
                        token_id = int(top_indices[0, k].item())
                        token_log_prob = float(top_log_probs[0, k].item())
                        next_seq = seq + [token_id]
                        next_alpha_seq = alpha_seq + [alpha.squeeze(0)]
                        next_finished = token_id == end_idx
                        candidates.append(
                            (score + token_log_prob, next_seq,
                             next_alpha_seq, h_next, c_next, next_finished)
                        )

                candidates.sort(key=lambda item: item[0], reverse=True)
                beams = candidates[:beam_size]

                still_active: list[tuple[float, list[int],
                                         list[Tensor], Tensor, Tensor, bool]] = []
                for beam in beams:
                    if beam[5]:
                        completed.append((beam[0], beam[1], beam[2]))
                    else:
                        still_active.append(beam)

                if not still_active:
                    break

            if completed:
                best_score, best_seq, best_alpha_seq = max(
                    completed, key=lambda item: item[0])
            else:
                best_score, best_seq, best_alpha_seq = max(
                    [(score, seq, alpha_seq)
                     for score, seq, alpha_seq, _h, _c, _f in beams],
                    key=lambda item: item[0],
                )

            _ = best_score
            # Drop the initial <START> token from returned sequence.
            token_ids = best_seq[1:]
            if token_ids and token_ids[-1] == end_idx:
                token_ids = token_ids[:-1]

            if best_alpha_seq:
                alpha_tensor = torch.stack(
                    best_alpha_seq[:len(token_ids)], dim=0)
            else:
                alpha_tensor = torch.empty(
                    (0, enc.size(1)), dtype=enc.dtype, device=device)

            batch_tokens.append(torch.tensor(
                token_ids, dtype=torch.long, device=device))
            batch_alphas.append(alpha_tensor)

        max_tokens_len = max((t.numel() for t in batch_tokens), default=0)
        max_alpha_len = max((a.size(0) for a in batch_alphas), default=0)

        padded_tokens = encoder_out.new_full(
            (len(batch_tokens), max_tokens_len),
            fill_value=end_idx,
        ).long()
        padded_alphas = encoder_out.new_zeros(
            (len(batch_alphas), max_alpha_len, encoder_out.size(1)))

        for i, token_tensor in enumerate(batch_tokens):
            if token_tensor.numel() > 0:
                padded_tokens[i, :token_tensor.numel()] = token_tensor

        for i, alpha_tensor in enumerate(batch_alphas):
            if alpha_tensor.size(0) > 0:
                padded_alphas[i, :alpha_tensor.size(0), :] = alpha_tensor

        return padded_tokens, padded_alphas
