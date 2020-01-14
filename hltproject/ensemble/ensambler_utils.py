
import logging
import os
import pickle
import math

from common_interface import model
from model_9.utils import *
from hltproject.score.score import compute_loss
from model9 import model_squad
from model9 import model_swag
    
logger = logging.getLogger ( __name__ )


def min_voting(valutaion_arrays):

    ris= np.asarray(valutaion_arrays)
    new_ris= []

    for xid in range(ris.shape[1]):     

        max_abs = 0
        for model_eval in ris[:, xid, :]:
            min_a = np.amin(model_eval)
            index = np.where(model_eval == min_a)

            if(min_a>=max_abs):
                max_abs=min_a
                index_abs= index
                a = model_eval

        new_ris.append(a)

    return new_ris

def max_voting(valutaion_arrays):

    ris= np.asarray(valutaion_arrays)
    new_ris= []

    for xid in range(ris.shape[1]):     

        max_abs = 0
        for model_eval in ris[:, xid, :]:
            max_a = np.amax(model_eval)
            index = np.where(model_eval == max_a)

            if(max_a>=max_abs):
                max_abs=max_a
                index_abs= index
                a = model_eval

        new_ris.append(a)

    return new_ris

def min_entropy(valutaion_arrays):

    ris= np.asarray(valutaion_arrays)
    new_ris= []

    #per each example
    for xid in range(ris.shape[1]):     

        #for each model probab in each example
        min_entropy = 99999999999999999999
        for model_eval in ris[:, xid, :]:

            tot=0
            for x in model_eval:
                tot = tot + (x*math.log2(x))

            tot=tot*(-1)

            if(tot<min_entropy):
                a = model_eval

        new_ris.append(a)

    return new_ris


class model_e(model):

    def __init__(self,modelli):
        self.modelli = modelli
        

    def train(self,train_set, validation_set):

        for modello in self.modelli:
            modello.train( train_set, vallidation_set )

    def evaluate(self,dataset,combination="mean"):

        risultati = []

        for modello in modelli:
            print("-")
            risultati.append(modello.evaluate(dataset))

        #with open('array_evalutation.pkl', 'rb') as f:
         #   risultati = pickle.load(f)

        
        if combination == "mean":
            return np.mean(risultati, axis=0)

        elif combination == "max":
            return np.asarray(max_voting(risultati))

        elif combination == "min":
            return np.asarray(min_voting(risultati))

        elif combination == "min_entropy":
            return np.asarray(min_entropy(risultati))

        return np.mean(risultati, axis=0)
