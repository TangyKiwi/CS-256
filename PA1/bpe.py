import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.pattern = {}
        self.token_to_id = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        
        words = text.split()
        vocab = collections.Counter(
            [" ".join(list(word)) for word in words]
        )

        chars = sorted(set(text))
        self.token_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
        }
        for c in chars:
            if c not in self.token_to_id:
                self.token_to_id[c] = len(self.token_to_id)
        self.vocab = {i: c for c, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<PAD>"]
        self.unk_id = self.token_to_id["<UNK>"]

        next_id = len(self.vocab)
        num_merges = vocab_size - next_id

        for i in range(num_merges):
            stats = get_stats(vocab)
            if not stats:
                break

            best = max(stats, key=stats.get)
            new_symbol = "".join(best)
            vocab = merge_vocab(best, vocab)

            self.merges[best] = new_symbol
            self.token_to_id[new_symbol] = next_id
            self.vocab[next_id] = new_symbol
            next_id += 1

            if verbose:
                print(f'Merge {i+1}/{num_merges}: {best} -> {new_symbol}')

    def decode(self, ids):
        return "".join([self.vocab[i] for i in ids])

    def encode(self, text):
        tokens = list(text)

        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            merge_candidates = [pair for pair in pairs if pair in self.merges]

            if not merge_candidates:
                break

            best_pair = merge_candidates[0]
            new_token = self.merges[best_pair]

            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

