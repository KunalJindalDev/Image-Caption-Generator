import torch.nn as nn

from src.encoder import EncoderCNN
from src.decoder import DecoderLSTM


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout=0.0):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderLSTM(
            vocab_size,
            embed_size,
            hidden_size,
            num_layers,
            dropout,
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)
