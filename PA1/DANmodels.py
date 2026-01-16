import torch
from torch import nn
import torch.nn.functional as F

from sentiment_data import WordEmbeddings

class DAN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, embeddings: WordEmbeddings):
        super().__init__()
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a list of word indices

        # frozen = True by default
        embedded_layer = self.embeddings.get_initialized_embedding_layer()

        # need to adjust average calculation to account for PAD token
        embedded_sentences = embedded_layer(x)
        sentence_embeddings = embedded_sentences.mean(dim=1)  # Shape: (batch_size, embedding_dim)

        hidden = F.relu(self.fc1(sentence_embeddings))  # Shape: (batch_size, hidden_dim)
        output = self.fc2(hidden)  # Shape: (batch_size, 2)
        log_probs = self.log_softmax(output)  # Shape: (batch_size, 2)
        return log_probs