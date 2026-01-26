import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import WordEmbeddings
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, embeddings: WordEmbeddings):
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
        
        return torch.LongTensor(indices), example.label

def DAN_collate_fn(batch):
    sentences, labels = zip(*batch)

    lengths = [len(s) for s in sentences]
    max_len = max(lengths)

    padded_sentences = []
    for s in sentences:
        pad_len = max_len - len(s)
        padded = torch.cat([
            s,
            torch.zeros(pad_len, dtype=torch.long)  # PAD = 0
        ])
        padded_sentences.append(padded)

    X = torch.stack(padded_sentences)
    y = torch.tensor(labels)

    return X, y

class DAN(nn.Module):
    def __init__(self, hidden_size: int, embeddings: WordEmbeddings):
        super().__init__()
        self.embeddings = embeddings.get_initialized_embedding_layer()
        self.embedding_dim = embeddings.get_embedding_length()

        self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded_sentences = self.embeddings(x)

        mask = (x != 0).unsqueeze(-1)
        masked_embeddings = embedded_sentences * mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        sentence_embeddings = sum_embeddings / mask.sum(dim=1).clamp(min=1)

        x = F.relu(self.fc1(sentence_embeddings))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x