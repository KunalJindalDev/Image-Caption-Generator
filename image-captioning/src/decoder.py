import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        logits = self.linear(hiddens)
        return logits[:, 1:, :]

    @torch.no_grad()
    def generate(self, features, max_len=50, start_idx=1, end_idx=2, idx_to_word=None):
        device = features.device
        batch_size = features.size(0)

        hidden_states = None

        # Step 0 mirrors training: run one LSTM step with image features only.
        image_inputs = features.unsqueeze(1)
        _, hidden_states = self.lstm(image_inputs, hidden_states)

        # Step 1 starts from the START token embedding.
        current_tokens = torch.full(
            (batch_size,), start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated_tokens = []

        for _ in range(max_len):
            inputs = self.embed(current_tokens).unsqueeze(1)
            hiddens, hidden_states = self.lstm(inputs, hidden_states)
            logits = self.linear(hiddens[:, -1, :])
            predicted = logits.argmax(dim=1)

            # Keep finished sequences pinned to END.
            predicted = torch.where(
                finished, torch.full_like(predicted, end_idx), predicted)
            generated_tokens.append(predicted)

            finished = finished | (predicted == end_idx)
            if torch.all(finished):
                break

            current_tokens = predicted

        if not generated_tokens:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)

        stacked_tokens = torch.stack(generated_tokens, dim=1)

        if not hasattr(self, "_printed_generation_debug"):
            self._printed_generation_debug = False

        if not self._printed_generation_debug and stacked_tokens.size(0) > 0:
            preview_ids = stacked_tokens[0, :5].tolist()
            if callable(idx_to_word):
                preview_words = [idx_to_word(int(token_id)) for token_id in preview_ids]
            elif isinstance(idx_to_word, dict):
                preview_words = [idx_to_word.get(int(token_id), "<UNK>") for token_id in preview_ids]
            else:
                preview_words = [str(token_id) for token_id in preview_ids]

            print(f"Generation debug - first 5 predicted token IDs: {preview_ids}")
            print(f"Generation debug - first 5 predicted words: {preview_words}")
            self._printed_generation_debug = True

        return stacked_tokens


class DecoderRNN(DecoderLSTM):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__(vocab_size, embed_size, hidden_size, num_layers, dropout=0.0)
