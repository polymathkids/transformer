# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        raise Exception("Implement me")
        """
        we have control over how to set up the embeddings
        embeddings-> vocab_size x d_model
        positional_encodings -> d_model x num_positions (20)

        we have a little bit of freedom in what we do
        you can initialize >= 1 TransformerLayer here

        Another FFNN/sequential-type layer (linear/ReLU/softmax/linear)
        - linear -> relu -> linear -> softmax (this approach should work well)


        !! Remember the last linear layer should have a output size of num_classes !!
        maybe nn.LayerNorm- can smooth out values in the input tensor by reformatting them- can up the accuracy if you are having trouble in results 
        """

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        raise Exception("Implement me")
        """
        embeddings -> positional_encodings -> TransformerLayer -> (TransformerLayer optional) -> ... -> FFNN/Seuqential -> return
        """


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        raise Exception("Implement me")
        """
        need a query, key, and value

        self.d_model =  between 64 to 128 to start, or try 20 (sequence length) to start
        self.d_internal =
        self.query = nn.Linear(d_model, d_internal)
        self.key = nn.Linear(d_model, d_internal)
        self.value = nn.Linear(d_model, d_internal?)

        for the above, we must have another linear layer to get from d_internal back to d_model in order to feed it into
        the sequential part

        have a little sequential model that you can label as a sequential model if you know how to do so (nn.sequential)
        do a sequential model here with Linear -> ReLU -> Linear

        Here the linear layers will be d_model x d_model
        or have the output of the first match up to the input of the second
        Linear(d_model, param) -> ReLU -> Linear(param, d_model)
        param might as well just be set to d_model.  Smaller not helpful, larger maybe.
        """

    def attention(self, query, key, value):

    # step 1:  multiply the query and the key together (torch.matmul or query @ key.T)
    # step 2:  divide the result by sqrt(d_internal) to prevent the values from exploding - MIGHT not be necessary
    # step 3:  apply a softmax to the result by the value (torch.nn.functional.softmax)
    # step 4:  multiply the result by the value (torch.matmul(softmax_result, v) or softmax_result @ v)

    def FFNN(self, attention):

    # linear -> relu -> linear

    def forward(self, input_vecs):
        raise Exception("Implement me")
        """
        query = self.query(input_vecs)
        key = self.key(input_vecs)
        value = self.value(input_vecs)

        attention = self.attention(query, key, value)
        return self.FFNN(attention)

        use matrix multiplication from torch, might have to divide by sqrt(d_internal)

        map = torch.matmul(query, key.transpose(1,0))
        we CAN softmax on the map, but don't necessarily have to nn.functional.softmax

        then we have to convert the map to the dimensionality of d_model - attention_output
        residual connection - adding the attention_output to the input vecs

        feed it through our sequential model - sequential_output
        residual connections - adding the sequential_output to the first residual_output, getting us the second residual_output
        return residual_output_2 (and maybe the map if we feel like we need it)
        """


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    raise Exception("Not fully implemented yet")
    """
    initialize stuff like d_model, d_internal, num_layers, epochs, learning rate, etc.

    I always recommend using nn.NLLoss(). NLLoss lets you play around with different dimensions
    This makes some things easier as you don't have to use a 1d vector
    Doesn't do the softmax for you, so lets you do it when you want
    Crossentropy covers the softmax part
    """
    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    model = Transformer(...)  # vocab_size, num_positions, d_model, d_internal, num_classes, num_layers
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parfameters(), lr=1e-4)

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            # model.zero_grad()
            """
            format the input and output into tensors
            (output, attn_values) = model(input)
            loss = loss_fcn(output, labels)
            """
            loss = loss_fcn(...)  # TODO: Run forward and compute loss
            # loss.backward()
            # optimizer.step()
            loss_this_epoch += loss.item()
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
