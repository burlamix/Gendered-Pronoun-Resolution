
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import copy
import math
import random

from hltproject.dataset_utils.parsing import parse_embeddings_dataset
import collections

import tqdm
import logging
import logging.config
import hltproject.utils.config as cutils

import itertools

logging.config.dictConfig(
    cutils.load_logger_config_file())
logger = logging.getLogger ( __name__ )

_MAX_EPOCHS = 1000
_MINIBATCH_SIZE = 100
_EARLY_STOPPING_EPOCHS = 3

#DEBUG
# _MAX_EPOCHS = 100
# _MINIBATCH_SIZE = 10
# _DATASET_LEN = 20

# Object that encapsulates neural network input, label and original sentence id
NNIO = collections.namedtuple ('NNIO', ['id', 'input', 'label'])

class BaselineNet (nn.Module):

    def __init__(self, input_dim, ):
        super(BaselineNet, self).__init__()
        self.fc1 = nn.Linear ( input_dim, 512 )
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)
        self.softmax = nn.Softmax ( dim=1 )

    def forward(self, x):
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        x = self.softmax (x)
        return x

def get_inputs_and_label ( sent, augment=False ):
    inputs = []
    emb_A = sent.embeddings[ sent.A_tok_off ]
    emb_B = sent.embeddings[ sent.B_tok_off ]
    emb_P = sent.embeddings[ sent.pron_tok_off ]

    if not augment:
        inputs = np.hstack ( (emb_P, emb_A, emb_B ) )
    else:
        emb_PA = emb_P * emb_A
        emb_PB = emb_P * emb_B
        emb_AB = emb_A * emb_B
        emb_PP = emb_P * emb_P
        inputs = np.hstack ( ( emb_P, emb_A, emb_B, emb_PA, emb_PB, emb_AB-emb_PP ) )

    
    label = 0 if sent.A_coref else 1 if sent.B_coref else 2
    # logger.debug ("sentence inputs shape {}".format( inputs.shape ))
    # logger.debug ("{} {} {}".format(sent.A_coref, sent.B_coref, label))
    return inputs, label

def sentence_to_nn_io ( sent, augment ):
    vec, label = get_inputs_and_label ( sent, augment )
    id = sent.id
    # logger.debug ("sentence_to_nn_io sentence {} - {} ({}/{})".format (sent.id, sent.tokens[:5], sent.A_coref, sent.B_coref))
    # logger.debug ("                  input shape {}".format (vec.shape) )
    # logger.debug ("                  label {}".format (label) )
    return NNIO ( id, vec, label )


def load_embeddings_minibatch ( dataset, size=0, shuffle=False ):
    minibatch_inputs_list = []
    minibatch_labels_list = []
    minibatch_ids_list = []
    count_iterator = range(len(dataset)) if size <= 0 else range(size)
    for i in count_iterator:
        nn_io = dataset[i] if not shuffle else dataset[ random.randint (0, len(dataset)-1) ]
        sent_inputs, sent_label = nn_io.input, nn_io.label
        minibatch_inputs_list.append ( sent_inputs )
        minibatch_labels_list.append ( sent_label )
        minibatch_ids_list.append (nn_io.id)

    inputs = np.vstack ( minibatch_inputs_list )

    # if shuffle:
    #     print ("chosen ids",minibatch_ids_list)
    # logger.debug (" load embedding minibatch ids: {}".format (minibatch_ids_list) )
    # logger.debug (" load embedding minibatch labels: {}".format (minibatch_labels_list) )
    # input ()
    
    return minibatch_ids_list, torch.as_tensor (inputs, dtype=torch.float), torch.as_tensor ( minibatch_labels_list )

@torch.no_grad ()
def compute_validation_loss ( validation_dataset, net, criterion ):
    _, inputs, labels = load_embeddings_minibatch ( validation_dataset )

    outputs = net( inputs )
    loss = criterion(outputs, labels)

    return outputs, loss.item()
        

def train_network ( net, train_dataset, validation_dataset ):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    number_of_weights_updates_per_epoch = len(train_dataset) // _MINIBATCH_SIZE

    validation_loss = 0.
    training_loss = 0.
    epoch = 0
    not_decreasing_epochs = 0
    best_model = None
    best_validation_loss = math.inf

    with tqdm.tqdm ( range(_MAX_EPOCHS), unit="epoch" ) as progress_bar:
        progress_bar.set_postfix_str ("train loss: ? valid loss: ?")
        while epoch < _MAX_EPOCHS and not_decreasing_epochs<_EARLY_STOPPING_EPOCHS:
            epoch += 1
            progress_bar.update ()
            
            training_loss = 0.0

            for _ in range ( number_of_weights_updates_per_epoch ) :
                _, inputs, labels = load_embeddings_minibatch ( train_dataset, _MINIBATCH_SIZE, shuffle=True )
                
                optimizer.zero_grad()
                outputs = net( inputs )
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                training_loss += loss.item()
            
            _, validation_loss = compute_validation_loss ( validation_dataset, net, criterion )
            training_loss = training_loss/number_of_weights_updates_per_epoch

            if validation_loss < best_validation_loss:
                not_decreasing_epochs = 0
                best_validation_loss = validation_loss
                best_model = copy.deepcopy ( net )
            else:
                not_decreasing_epochs += 1

            progress_bar.set_postfix_str ("train loss: {:.3f} valid loss: {:.3f}".format(training_loss, validation_loss))
            
    return epoch, training_loss, best_validation_loss, best_model
            
    
def compute_predictions ( train_fname, validation_fname, test_fname, augment, output_fname ):
    
    #get embeddings size from the first train sentence
    first_sent = next ( parse_embeddings_dataset(train_fname) )
    embeddings_size = len(first_sent.embeddings[0])
    logger.info ("embeddings size: {}".format( embeddings_size ))
    
    logger.info ("loading train dataset...")
    weights = [0,0,0]
    train_dataset = []
    for sent in tqdm.tqdm (parse_embeddings_dataset(train_fname), unit="sentences"):
        input_vector = sentence_to_nn_io (sent, augment)
        train_dataset.append (input_vector)
        label = 0 if sent.A_coref else 1 if sent.B_coref else 2 
        weights[label] += 1
    logger.info ("loaded  train dataset: {} sentences".format(len(train_dataset)))
    weights = [ weights[0]/len(train_dataset), weights[1]/len(train_dataset), weights[2]/len(train_dataset) ]
    logger.info ("class distribution: A={}, B={}, N={}".format(weights[0],weights[1],weights[2]))

    logger.info ("loading validation dataset...")
    dataset_iterator = tqdm.tqdm ( parse_embeddings_dataset (validation_fname ), unit="sentences" )
    validation_dataset = [ sentence_to_nn_io (sent, augment) for sent in dataset_iterator ]
    logger.info ("loaded  validation dataset: {} sentences".format(len(validation_dataset)))
    
    # if not augment: input size = 3 embeddings (pronoun, A, B) 
    #     if augment: input size = 6 embeddings (pronoun, A, B, PA, PB, AB-PP)
    input_dim = 3*embeddings_size if not augment else 6*embeddings_size
    net = BaselineNet ( input_dim )
    
    logger.info ("Starting training phase.")
    epoch, training_loss, validation_loss, net = train_network ( net, train_dataset, validation_dataset )
    logger.info ("Finished training in {} epochs. train loss: {:.5f} valid loss: {:.5f}".format(epoch, training_loss, validation_loss))

    # free memory (?)
    del validation_dataset
    del train_dataset

    logger.info ("loading test dataset...")
    dataset_iterator = tqdm.tqdm ( parse_embeddings_dataset (test_fname ), unit="sentences" )
    ids, inputs, _ = load_embeddings_minibatch ( [ sentence_to_nn_io (sent, augment) for sent in dataset_iterator ] )
    logger.info ("loaded  test dataset: {} sentences".format(len(ids)))

    logger.info ("computing test scores...")
    with torch.no_grad ():
        outputs = net (inputs)

    logger.info ("writing out results...")
    with open (output_fname, "w") as fout:
        print ("ID,A,B,NEITHER", file=fout)
        for sent_id, scores in zip (ids, outputs):
            prob_A, prob_B, prob_N = scores[0].item(), scores[1].item(), scores[2].item()
            print (",".join ([sent_id, str(prob_A), str(prob_B), str(prob_N)]), file=fout)

            
    
            

