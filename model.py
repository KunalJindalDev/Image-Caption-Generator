import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Extract features using pretrained ResNet-101 [cite: 10, 13]
        # We use weights=ResNet101_Weights.DEFAULT for modern PyTorch versions
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
        # Remove the final fully connected classification layer 
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Linear layer to map the 2048-dim pooled feature vector to the embedding space 
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # Forward pass through ResNet
        features = self.resnet(images)                     # Shape: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)     # Shape: (batch_size, 2048)
        
        # Create the single global average-pooled feature vector to seed the decoder 
        features = self.bn(self.linear(features))          # Shape: (batch_size, embed_size)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Standard LSTM without attention 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Exclude the `<end>` token from captions so we don't pass it as an input sequence
        # captions shape: (batch_size, max_seq_length)
        embeddings = self.embed(captions[:, :-1])
        
        # Concatenate the image feature vector at the start of the sequence
        # This seeds the LSTM with the image context before generating word by word [cite: 12, 16]
        # features shape: (batch_size, embed_size) -> (batch_size, 1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # Pass through the LSTM
        hiddens, _ = self.lstm(embeddings)                 # Shape: (batch_size, max_seq_length, hidden_size)
        outputs = self.linear(hiddens)                     # Shape: (batch_size, max_seq_length, vocab_size)
        
        return outputs
    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs