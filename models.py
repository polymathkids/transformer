# models.py

import argparse
import random
import numpy as np
from data import *
from models import *
from utils import *
from typing import List
import torchvision

from transformers import pipeline, AutoTokenizer, BertConfig, BertModel, BartConfig, BartForConditionalGeneration, BartModel, TrainingArguments, Trainer


def initialize_seq2seq_model(vocab_size: int):
    """
    :param vocab_size: The size of the vocabulary to use
    :return: A randomly initialized BartForConditionalGeneration model, with relatively small parameter sizes compared
    to what they are by default. You can modify these to experiment if you'd like, but it's outside the scope of
    what's intended in the assignment.
    """
    # The hyperparameters of the BART model are defined here
    # They are smaller than any of the standard BART models,
    # so that we can train it form scratch on our semantic parsing dataset
    config = BartConfig(
        vocab_size=vocab_size,
        max_position_embeddings=100, # was 1024
        encoder_layers=4, # was 12
        encoder_ffn_dim=256, # was 4096
        encoder_attention_heads=4, # was 16
        decoder_layers=4, # was 12
        decoder_ffn_dim=256, # was 4096
        decoder_attention_heads=8, # was 16
        d_model=128) # was 1024

    # The BART model with random weights gets created here
    model = BartForConditionalGeneration(config)
    return config, model


def train_seq2seq_model(model, train_dataset, val_dataset, args):
    """
    :param model: a randomly initialized seq2seq model sharing the BART architecture
    :param train_dataset: the preprocessed train dataset
    :param val_dataset: the preprocessed validation (dev) dataset
    :param args: args bundle from main
    :return: nothing; trains the seq2seq model and updates its parameters in-place
    """
    # We define a set of arguments to be passed to the trainer
    # Most of the arguments are training related hyperparameters
    # some are about storing checkpoint and tensorboard logs
    training_args = TrainingArguments(        
        output_dir=args.model_save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        learning_rate=5e-04, # default seems to be 5e-05
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    # The trainer is a class that handles all aspects of the
    # training for us. We pass to it the model, dataset and other
    # hyperparameters. We don't have to worry about writing the
    # training loop or dealing with checkpointing, logging etc..
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    # We call the train function on the trainer
    # This trains the model we passed through on the dataset
    # we passed through
    trainer.train()


def score_sequence(pred_to_first_eos, gold_labels) -> (int, int, int):
    """
    Evaluates the given sequence and returns the sufficient statistics for accuracy computation.
    DO NOT MODIFY THIS -- we are giving it to you here in models.py for convenience, but we will compute your accuracy
    with an unmodified version of it.
    :param pred_to_first_eos: predicted tokens (real tokens, not indexed) up to and including an <EOS> token
    :param gold_labels: the gold labels (which also include EOS)
    :return: a tuple of exact_match (1 or 0), the token-level recall (the fraction of tokens in the gold that are matched
    by a corresponding token in the prediction), and the total number of tokens
    """
    exact_match = 0
    recall = 0
    total_toks = 0
    total_toks += len(gold_labels)
    if pred_to_first_eos == gold_labels:
        exact_match += 1
    print("Pred: " + ' '.join(pred_to_first_eos))
    print("Gold: " + ' '.join(gold_labels))
    for j in range(0, len(gold_labels)):
        if j < len(pred_to_first_eos) and pred_to_first_eos[j] == gold_labels[j]:
            recall += 1
    return (exact_match, recall, total_toks)


def score_decoded_outputs(all_preds, indexer, exs):
    """
    Prints two metrics:
    1. Token-level recall: what fraction of the gold tokens are exactly predicted *at the same position* by
    the model.
    2. Exact match: how often the entire sequence exactly matches the gold standard.
    :param all_preds: list of the model's predictions, must be shorter than or equal to exs in size
    :param indexer:
    :param exs: the examples
    :return:
    """
    (top1_exact_match, top1_recall, top1_total_toks) = (0, 0, 0)
    for i in range(0, len(all_preds)):
        gold_labels = [indexer.get_object(j) for j in exs[i]['labels'] if j >= 0]
        one_best = all_preds[i]
        (top1_em, top1_rec, top1_tt) = score_sequence(one_best, gold_labels)
        top1_exact_match += top1_em
        top1_recall += top1_rec
        top1_total_toks += top1_tt
    print("Recall: " + repr(top1_recall) + "/" + repr(top1_total_toks) + " = " + repr(top1_recall / top1_total_toks))
    print("Exact Match: " + repr(top1_exact_match) + "/" + repr(len(all_preds)) + " = " + repr(top1_exact_match / len(all_preds)))


def pred_indices_to_prediction(raw_pred_indices, indexer):
    """
    :param raw_pred_indices: the raw output of the model
    :param indexer
    :return: The decoded sequence *stopping at the first occurrence of EOS*
    """
    pred_labels = [indexer.get_object(id.item()) for id in raw_pred_indices]
    # Find the first EOS token or take the whole sequence if there's no EOS
    first_eos_tok = pred_labels.index(EOS_SYMBOL) if EOS_SYMBOL in pred_labels else len(pred_labels) - 1
    return pred_labels[0:first_eos_tok + 1] # include the EOS in the return


def decode_basic(model, indexer, exs, num_exs=-1):
    """
    Basic decoding method to show how to use .generate() from a HuggingFace model to get outut
    :param model:
    :param indexer:
    :param exs:
    :param num_exs: -1 if we should use all the examples, otherwise a small number to allow decoding on fewer examples
    :return:
    """
    all_example_preds = []
    num_exs_to_use = min(num_exs, len(exs)) if num_exs > 0 else len(exs)
    for i in range(0, num_exs_to_use):
        ex_length = sum(exs[i]['attention_mask'])
        dev_input_tensor = torch.tensor([exs[i]['input_ids'][0:ex_length]], dtype=torch.long)
        # You can increase this to run "real" beam search
        beam_size = 1
        # The generate method runs decoding with the specified set of
        # hyperparameters and returns a list of possible sequences
        output_ids = model.generate(dev_input_tensor, num_beams=beam_size, max_length=65, early_stopping=True, num_return_sequences=beam_size)
        # [0] extracts the first candidate in the beam for the simple decoding method
        one_best = pred_indices_to_prediction(output_ids.data[0][1:], indexer) # this and next line replaced with for loop in oracles
        all_example_preds.append(one_best)
    return all_example_preds

#piggybacks off generate function in hugging face
# in oracle decoding- turns up the beam size and look at what these different ouputs are and then see alternatives the model produces
def decode_oracle(model, indexer, exs, num_exs):
    # Same as decode_basic but returns the oracle prediction
    '''
    Tips:
    increase beam size (10ish would be good)
    90% of code will be decode_basic
    use gold labels to help cheat-> return the correct index
    -will have 2-3 additional lines than decode basic if done efficiently in the one-best area
     (use a combo like getting the correct dot product and then getting p-values)
     iterate through the beam size and check the scores along the way and keep track of best one and then select that one
    otherwise- use a for-loop to get best score
    '''
    raise Exception("Implement me")

# you want to rerank those outputs by looking through htem by a set of rules- will follow the code in model.generate quite closely
#the amount of code you need to write is not that high- will lean heavily on decode_basic
# if you are under 50%, you will need to up the beam size- may need to train a couple times to get a good model
# the actual beam decoding- the higher you set the beam size, the better the oracle will be
# the more options you have, the higher chance you have of getting it

def decode_fancy(model, indexer, exs, num_exs):
    # Same as decode_basic but returns a reranked prediction.
    '''
    Tips
    similar/ same beam size as decode oracle
    everything through model.generate() will be same as decode oracle
    requires a few more lines than decode-oracle after that
    overlal, differences between the two oracles should be minimal- mostly in reranking in the for loop
    will need to flip a few signs and get a few matrices in different orders- but not much to do
    -score- use score function model or create your own
    '''
    raise Exception("Implement me")


# 'east' doesn't actually exist
const_list = ['new', 'north', 'south', 'west', 'east', 'alabama', 'alaska', 'arizona', 'arkansas', 'california',
    'colorado', 'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois',
    'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
    'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'hampshire', 'jersey', 'mexico', 'york', 'carolina', 'dakota',
    'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode', 'island', 'tennessee', 'texas', 'utah', 'vermont', 'virginia',
    'washington', 'virginia', 'wisconsin', 'wyoming']


##########
#PART 1
########

#Transformer Layer
# This layer should follow the format discussed in class:
# (1) self-attention (single- headed is fine; you can use either backward-only or bidirectional attention);
# (2) residual connection;
# (3) Linear layer, nonlinearity, and Linear layer;
# (4) final residual connection.
# With a shallow network like this, you likely don’t need layer normalization, which is a bit more complicated to implement.
# Because this task is relatively simple, you don’t need a very well-tuned architecture to make this work.
# You will implement all of these components from scratch.
# You will want to form queries, keys, and values matrices with linear layers,
# then use the queries and keys to compute attention over the sentence,
# then combine with the values.
# You’ll want to use matmul for this purpose, and you may need to transpose matrices as well.
# Double-check your dimensions and make sure everything is happening over the correct dimension.

def TransformerLayer():


class TransformerLayer(nn.Module):
    """
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, ReLU nonlinearity, another matrix multiplication

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.

    Strategy:  feed in positional encoding into a self attention calculation that adds the output to the input, feeds that into a FFN, takes the output of that and adds back in the original encodings.
Then the Transformer takes this layer's outputs, runs it through a linear layer to collapse down to 3 classes and then outputs the log_softmax.
2) Have a look at the outputs from self-attention prior the FF layer and make sure those values are sane.
3) Did you use the correct labels for your training? Note: the labels should be the indexes to characters if you use NLLLoss, not a one hot vector
4) Make sure you are applying the softmax across the correct dimension.

    """

    def __init__(self, in_size = 20, n_layer1, n_layer2, num_classes = 3):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.
        """
        super(TransformerLayer, self).__init__()

        # instantiate layers
        self.nonlin = nn.ReLU()  # try nn.ReLU(), nn.Tanh(), nn.ReLU6()
        self.linear1 = nn.Linear(in_size, n_layer1)
        self.linear2 = nn.Linear(n_layer1, num_classes)


        # initialize weights
        nn.init.kaiming_uniform_(self.linear1.weight)
        nn.init.kaiming_uniform_(self.linear2.weight)


    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """

        x = self.linear2(self.nonlin(self.linear1(x)))

        return x