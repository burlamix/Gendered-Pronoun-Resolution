
import logging
import os

from common_interface import model
from model_9.utils import *
from model9 import model9
#from model_7.Step1_preprocessing import original_notebook_preprocessing




class model_e(model):

    def __init__(self,modelli):
        self.modelli = modelli
        

    def train(train_set, validation_set):

        for modello in self.modelli:
            modello.train( train_set, vallidation_set )



    def evaluate(dataset, modelli_pesi):

        risultati = []

        for modello in modelli:
            risultati.append(modello.evaluate(test_path))

        #qui decido come fare ensambling media semplice?
        final_preds = np.mean(risultati, axis=0)

        return final_preds


if __name__ == "__main__":

    '''
    test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
    dev_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv"
    val_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv"
    '''
    #per trainare e testare piu velocemente, sono solo 5 esempi
    test_path = "../datasets/gap-light.tsv"
    dev_path = "../datasets/gap-light.tsv"
    val_path = "../datasets/gap-light.tsv"


    modelli = []

    modelli.append(model9("model_9/weights"))
    modelli.append(model9("model_9/weights"))
    modelli.append(model9("model_9/weights"))


    model_e_inst = model_e(modelli)


    final_predictions = model_e_inst.evaluate(test_path)

