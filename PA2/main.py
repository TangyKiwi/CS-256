import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder, FeedForwardClassifier, TransformerDecoder
from utilities import Utilities

from tqdm import tqdm
import time
import argparse

seed = 42

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

g = torch.Generator()
g.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run project assignment in parts or all.')
    parser.add_argument('--part', type=str, required=True, help='Part to run (test, 1, 2, 3, all)')

    # Parse the command-line arguments
    args = parser.parse_args()
    PART = args.part

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if PART in ['test', '1', 'all']:
        encoder = TransformerEncoder(
                vocab_size=tokenizer.vocab_size,
                block_size=block_size,
                embed_size=n_embd,
                num_heads=n_head,
                num_layers=n_layer
            ).to(device)
    
    if PART in ['test', '2', '3', 'all']:
        decoder = TransformerDecoder(
                vocab_size=tokenizer.vocab_size,
                block_size=block_size,
                embed_size=n_embd,
                num_heads=n_head,
                num_layers=n_layer,
                hidden_dim=n_hidden
            ).to(device)
    
    if PART in ['test', '3', 'all']:
        window_size = 8
        decoder_3 = TransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            block_size=block_size,
            embed_size=n_embd,
            num_heads=n_head,
            num_layers=n_layer,
            hidden_dim=n_hidden,
            window_size=window_size
        ).to(device)
        decoder_3.load_state_dict(decoder.state_dict())  # Initialize with the same weights as the original decoder

    if PART in ['test', 'all']:
        sanity_sentence_short = "This is a test sentence for sanity check, it has almost thirty words in it to fill the majority of the attention map graph."
        sanity_sentence_long = "This is a really long sentence that is meant to for the sanity test check, it has more than thirty two words in it so that we can see how the attention map looks when the sentence length exceeds the given block size."
        print("Running sanity checks for attention maps ...")
        print("Sanity Check Test w/ Short Sentence:")
        print(sanity_sentence_short)
        print("Sanity Check Test w/ Long Sentence:")
        print(sanity_sentence_long)

        encoder.eval()
        utils = Utilities(tokenizer, encoder)

        print("Encoder Sanity Check Test w/ Short Sentence:")
        utils.sanity_check(
            sanity_sentence_short, 
            block_size,
            "short"
        )

        print("Encoder Sanity Check Test w/ Long Sentence:")
        utils.sanity_check(
            sanity_sentence_long, 
            block_size,
            "long"
        )

        decoder.eval()
        utils = Utilities(tokenizer, decoder)

        print("Decoder Sanity Check Test w/ Short Sentence:")
        utils.sanity_check(
            sanity_sentence_short, 
            block_size,
            "short"
        )

        print("Decoder Sanity Check Test w/ Long Sentence:")
        utils.sanity_check(
            sanity_sentence_long, 
            block_size,
            "long"
        )

        decoder_3.eval()
        utils = Utilities(tokenizer, decoder_3)

        print("Decoder w/ Local Window Attention Sanity Check Test w/ Short Sentence:")
        utils.sanity_check(
            sanity_sentence_short, 
            block_size,
            "short_3"
        )

        print("Decoder w/ Local Window Attention Sanity Check Test w/ Long Sentence:")
        utils.sanity_check(
            sanity_sentence_long, 
            block_size,
            "long_3"
        )

    if PART in ['1', 'all']:
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

        classifier = FeedForwardClassifier(
            encoder=encoder,
            input_dim=n_input,
            hidden_dim=n_hidden,
            output_dim=n_output
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
        
    if PART in ['2', '3', 'all']:
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True, generator=g)

        def make_lm_loader(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            dataset = LanguageModelingDataset(tokenizer, text, block_size)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=g)

        test_obama_loader = make_lm_loader("speechesdataset/test_LM_obama.txt")
        test_wbush_loader = make_lm_loader("speechesdataset/test_LM_wbush.txt")
        test_hbush_loader = make_lm_loader("speechesdataset/test_LM_hbush.txt")

    if PART in ['2', '3', 'all']:
        lm_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    if PART in ['3', 'all']:
        lm_optimizer_3 = torch.optim.Adam(decoder_3.parameters(), lr=learning_rate)

     # for the classification  task, you will train for a fixed number of epochs like this:

    if PART in ['1', 'all']:
        epoch_summaries = []
        t0 = time.perf_counter()
        for epoch in range(epochs_CLS):
            epoch_start_time = time.perf_counter()
            for xb, yb in tqdm(train_CLS_loader, desc=f"Epoch {epoch + 1}/{epochs_CLS}"):
                xb, yb = xb.to(device), yb.to(device)

                # CLS training code here
                optimizer.zero_grad(set_to_none=True)

                logits = classifier(xb)
                loss = criterion(logits, yb)

                loss.backward()
                optimizer.step()
            train_acc = compute_classifier_accuracy(classifier, train_CLS_loader)
            epoch_compute_time = time.perf_counter() - epoch_start_time
            print(f"Epoch {epoch + 1}/{epochs_CLS}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.2f}%, Time: {epoch_compute_time:.2f}s")
            epoch_summaries.append({
                "epoch": epoch + 1,
                "loss": loss.item(),
                "train_acc": train_acc,
                "time": epoch_compute_time
            })
        total_train_time = time.perf_counter() - t0
        final_train_acc = compute_classifier_accuracy(classifier, train_CLS_loader)
        final_test_acc = compute_classifier_accuracy(classifier, test_CLS_loader)

        model_summary = {
            "vocab_size": tokenizer.vocab_size,
            "block_size": block_size,
            "embed_size": n_embd,
            "num_heads": n_head,
            "num_layers": n_layer,
            "n_input": n_input,
            "n_hidden": n_hidden,
            "n_output": n_output
        }

        for k, v in model_summary.items():
            print(f"{k}: {v}")

        training_summary = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs_CLS": epochs_CLS,
            "total_params": sum(p.numel() for p in classifier.parameters()),
            "trainable_params": sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        }

        for k, v in training_summary.items():
            print(f"{k}: {v}")

        for s in epoch_summaries:
            print(f"Epoch {s['epoch']}: Loss={s['loss']:.4f}, Train Accuracy={s['train_acc']:.2f}%, Time={s['time']:.2f}s")

        print(f"Total Training Time: {total_train_time:.2f}s")
        print(f"Final Train Accuracy: {final_train_acc:.2f}%")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    if PART in ['2', '3', 'all']:
        t0 = time.perf_counter()
        iter_summaries = []
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(tqdm(train_LM_loader, desc="Training LM", total=max_iters)):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            # LM training code here
            lm_optimizer.zero_grad(set_to_none=True)
            loss = decoder(xb, yb)
            loss.backward()
            lm_optimizer.step()

            if (i + 1) % eval_interval == 0 or (i + 1) == 1 or (i + 1) == max_iters:
                train_perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
                test_obama_perplexity = compute_perplexity(decoder, test_obama_loader, eval_iters)
                test_wbush_perplexity = compute_perplexity(decoder, test_wbush_loader, eval_iters)
                test_hbush_perplexity = compute_perplexity(decoder, test_hbush_loader, eval_iters)
                print(f"\nIteration {i + 1}/{max_iters}, Loss: {loss.item():.4f}, Train Perplexity: {train_perplexity:.2f}, Test Obama Perplexity: {test_obama_perplexity:.2f}, Test W. Bush Perplexity: {test_wbush_perplexity:.2f}, Test H. Bush Perplexity: {test_hbush_perplexity:.2f}")
                iter_summaries.append({
                    "iteration": i + 1,
                    "loss": loss.item(),
                    "train_perplexity": train_perplexity,
                    "test_obama_perplexity": test_obama_perplexity,
                    "test_wbush_perplexity": test_wbush_perplexity,
                    "test_hbush_perplexity": test_hbush_perplexity
                })
        total_lm_train_time = time.perf_counter() - t0
        final_train_perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
        final_test_obama_perplexity = compute_perplexity(decoder, test_obama_loader, eval_iters)
        final_test_wbush_perplexity = compute_perplexity(decoder, test_wbush_loader, eval_iters)
        final_test_hbush_perplexity = compute_perplexity(decoder, test_hbush_loader, eval_iters)

        for s in iter_summaries:
            print(f"Iteration {s['iteration']}: Loss={s['loss']:.4f}, Train Perplexity={s['train_perplexity']:.2f}, Test Obama Perplexity={s['test_obama_perplexity']:.2f}, Test W. Bush Perplexity={s['test_wbush_perplexity']:.2f}, Test H. Bush Perplexity={s['test_hbush_perplexity']:.2f}")

        print(f"Total LM Training Time: {total_lm_train_time:.2f}s")
        print(f"Final Train Perplexity: {final_train_perplexity:.2f}")
        print(f"Final Test Obama Perplexity: {final_test_obama_perplexity:.2f}")
        print(f"Final Test W. Bush Perplexity: {final_test_wbush_perplexity:.2f}")
        print(f"Final Test H. Bush Perplexity: {final_test_hbush_perplexity:.2f}")
        print(f"Total Parameters in Decoder LM: {sum(p.numel() for p in decoder.parameters())}")
        print(f"Trainable Parameters in Decoder LM: {sum(p.numel() for p in decoder.parameters() if p.requires_grad)}")

    if PART in ['3', 'all']:
        t0 = time.perf_counter()
        iter_summaries = []
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(tqdm(train_LM_loader, desc="Training LM w/ Local Window Attention", total=max_iters)):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            # LM training code here
            lm_optimizer_3.zero_grad(set_to_none=True)
            loss = decoder_3(xb, yb)
            loss.backward()
            lm_optimizer_3.step()

            if (i + 1) % eval_interval == 0 or (i + 1) == 1 or (i + 1) == max_iters:
                train_perplexity = compute_perplexity(decoder_3, train_LM_loader, eval_iters)
                test_obama_perplexity = compute_perplexity(decoder_3, test_obama_loader, eval_iters)
                test_wbush_perplexity = compute_perplexity(decoder_3, test_wbush_loader, eval_iters)
                test_hbush_perplexity = compute_perplexity(decoder_3, test_hbush_loader, eval_iters)
                print(f"\nIteration {i + 1}/{max_iters}, Loss: {loss.item():.4f}, Train Perplexity: {train_perplexity:.2f}, Test Obama Perplexity: {test_obama_perplexity:.2f}, Test W. Bush Perplexity: {test_wbush_perplexity:.2f}, Test H. Bush Perplexity: {test_hbush_perplexity:.2f}")
                iter_summaries.append({
                    "iteration": i + 1,
                    "loss": loss.item(),
                    "train_perplexity": train_perplexity,
                    "test_obama_perplexity": test_obama_perplexity,
                    "test_wbush_perplexity": test_wbush_perplexity,
                    "test_hbush_perplexity": test_hbush_perplexity
                })
        total_lm_train_time = time.perf_counter() - t0
        final_train_perplexity = compute_perplexity(decoder_3, train_LM_loader, eval_iters)
        final_test_obama_perplexity = compute_perplexity(decoder_3, test_obama_loader, eval_iters)
        final_test_wbush_perplexity = compute_perplexity(decoder_3, test_wbush_loader, eval_iters)
        final_test_hbush_perplexity = compute_perplexity(decoder_3, test_hbush_loader, eval_iters)

        for s in iter_summaries:
            print(f"Iteration {s['iteration']}: Loss={s['loss']:.4f}, Train Perplexity={s['train_perplexity']:.2f}, Test Obama Perplexity={s['test_obama_perplexity']:.2f}, Test W. Bush Perplexity={s['test_wbush_perplexity']:.2f}, Test H. Bush Perplexity={s['test_hbush_perplexity']:.2f}")

        print(f"Total LM Training Time: {total_lm_train_time:.2f}s")
        print(f"Final Train Perplexity: {final_train_perplexity:.2f}")
        print(f"Final Test Obama Perplexity: {final_test_obama_perplexity:.2f}")
        print(f"Final Test W. Bush Perplexity: {final_test_wbush_perplexity:.2f}")
        print(f"Final Test H. Bush Perplexity: {final_test_hbush_perplexity:.2f}")
        print(f"Total Parameters in Decoder LM: {sum(p.numel() for p in decoder_3.parameters())}")
        print(f"Trainable Parameters in Decoder LM: {sum(p.numel() for p in decoder_3.parameters() if p.requires_grad)}")

if __name__ == "__main__":
    main()
