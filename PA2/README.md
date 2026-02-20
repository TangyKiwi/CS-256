# PA 2

Project explores the implementatin of encoders and decoders from scratch, attempting
to predict, given a short segment from a speech by Obama, George W. Bush, or 
George H. Bush, which former president the segment is from.

`main.py` and `transformer.py` contain all relevant code pertaining to 
the assignment. 

## Usage
### Part 1
Implemented a transformer encoder with a feed forward neural network classifier
in `transformer.py`. You can run this by:
```
python main.py --part 1
```

### Part 2
Implemented a transformer decoder in `transformer.py`. This uses the same
base blocks as the encoder, but modified for causal masking. You can run this by:
```
python main.py --part 2
```

### Part 3
Implemented a sparse attention pattern into the transformer decoder using a local
window attention. You can run this by:
```
python main.py --part 3
```

### Sanity Test Checks
You can run all sanity test checks for parts 1, 2, and 3 by running:
```
python main.py --part test
```
All output will go into the respective `encoder_attn_maps` or `decoder_attn_maps`
folders with appropriate naming. Sanity test checks are done with a short sentence
and a long sentence. 

### All
You can run all parts + sanity test checks by running:
```
python main.py --part all
```

No other run configurations are supported. The `--part` parameter is required
to run. All parameters related to each model are hardcoded for the assignment.

## Output
All relevant training debug information is printed to the console, including
data load time, epoch-specific training accuracies, training time, and final 
accuracies / perplexities. All sanity check attention map outputs can be found 
in the respective `encoder_attn_maps` or `decoder_attn_maps` directories, which 
are automatically created during runtime if the directory does not exist. Contains 
`.png` files with a header of the model architecture `encoder` or `decoder`, followed
by the sentence type tested `short` or `long`, with part 3 being distinguished
with a `_3` footer. 8 attention maps are produced for each test, leading to 16
total per part.
