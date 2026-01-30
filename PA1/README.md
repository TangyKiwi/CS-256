# PA 1

Project explores the implementatin of a simple Bag of Words model, as well as 
varied implementations of a Deep Averaging Network model. 

`main.py`, `DANmodels.py`, and `bpe.py` contain all relevant code pertaining to 
the assignment.

## Usage
### Part 1
1a) Implemented DAN model with pretrained embeddings in `DANmodels.py`. Pretrained
embeddings are hardcoded to the `embeddings` variable in `main.py`. You can use
either `embeddings_50d` or `embeddings_300d` (default). You can 
run this by:
```
python main.py --model DAN
```

1b) Implemented DAN model without pretrained GloVe embeddings. You can run this 
by:
```
python main.py --model DAN_random
```

### Part 2
2a) Implemented DAN model using subword tokenization with the BPE algorithm in 
`bpe.py`. You can run this by:
```
python main.py --model DANBPE
```

No other run configurations are supported. All other parameters related to
each model are hardcoded for the assignment.

## Output
All relevant training debug information is printed to the console, including
data load time, epoch-specific training and dev accuracies, and specifically
for the `DANBPE` model, the time it took to train the BPE tokenizer. All graph
outputs can be found in the `out_accuracy` directory, which is automatically 
created during runtime if the directory does not exist. Contains `.png` files
with a header of the model `bow`, `dan`, `dan_random`, or `dan_bpe` along with
a tail of either `train_accuracy` or `dev_accuracy`.
