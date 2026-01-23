import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import WordEmbeddings
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, embeddings: WordEmbeddings):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        self.word_indexer = embeddings.word_indexer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        indices = [self.word_indexer.index_of(word) 
                   if self.word_indexer.index_of(word) != -1 
                   else self.word_indexer.index_of("UNK")
                   for word in example.words
                ]
        
        # add PAD handling logic 
        return torch.LongTensor(indices), example.label

class DAN(nn.Module):
    def __init__(self, hidden_size: int, embeddings: WordEmbeddings):
        super().__init__()
        self.embeddings = embeddings.get_initialized_embedding_layer()
        self.embedding_dim = embeddings.get_embedding_length()

        self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a list of word indices

        # need to adjust average calculation to account for PAD token?
        embedded_sentences = self.embeddings(x)
        sentence_embeddings = embedded_sentences.mean(dim=1)  # Shape: (batch_size, embedding_dim)

        x = F.relu(self.fc1(sentence_embeddings))  # Shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # Shape: (batch_size, 2)
        x = self.log_softmax(x)  # Shape: (batch_size, 2)
        return x