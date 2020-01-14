import logging
import os
import pandas as pd
import numpy as np

from common_interface import model
from model_9.utils import BertSwagRunner
from model_9.utils import SquadRunner
from model_9.utils import BERTSpanExtractor

from sklearn.metrics import log_loss

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)


from hltproject.score.score import compute_loss_df
from hltproject.score.score import compute_loss

#from dataset_utils import compute_loss_simo

logger = logging.getLogger ( __name__ )



class model_b(model):
    ''' wrapper for 9th place model
        code: https://github.com/rakeshchada/corefqa
        paper: https://arxiv.org/pdf/1906.03695.pdf
    '''
    def __init__(self):

        None

    def train(self, train_set, vallidation_set ):

        print("!")
        self.runner.train( train_set, vallidation_set, self.weight_path, n_splits=4)


    def evaluate(self, val_df ):

        return  self.runner.my_evaluate( val_df, self.weight_path, is_test=False)

    def fit(self, train_set , vallidation_set ):

        self.runner.train( train_set, vallidation_set, self.weight_path, n_splits=4)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"weight_path": self.weight_path}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self



class model_swag(model_b):

    def __init__(self,weight_path):

        self.runner = BertSwagRunner(None, None, None, num_train_epochs=1, bert_model='bert-large-uncased')
        self.weight_path = weight_path
        self.classes_ = [3]



class model_squad(model_b):

    def __init__(self,weight_path):

        self.runner = SquadRunner(None, None, None, num_train_epochs=1, bert_model='bert-large-uncased')
        self.weight_path = weight_path
        self.classes_ = [3]

class model_SpanExtractor(model_b):

    def __init__(self,weight_path):

        self.runner = BERTSpanExtractor(None, None, None,  bert_model='bert-large-uncased')
        self.weight_path = weight_path
        self.classes_ = [3]
#------------------------------------------------------------------



#UNIT TESTS
if __name__ == "__main__":

    ''' 
    test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
    dev_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv"
    val_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv"
    


    dev_path = "..ensemble/model_7_submissions/input/gap-development_Alice_Kate_John_Michael.tsv"
    val_path  = "..ensemble/model_7_submissions/input/gap-validation_Alice_Kate_John_Michael.tsv"
    test_path = "..ensemble/model_7_submissions/input/gap-test_Alice_Kate_John_Michael.tsv"


    test_path = "../datasets/gap-validation.tsv"
    dev_path = "../datasets/gap-test.tsv"
    val_path = "../datasets/gap-development.tsv"
    zxzx = "../datasets/gap-light.tsv"
    '''


    #per trainare e testare piu velocemente, sono solo 5 esempi

    dev_path1  = "../ensemble/model_7_submissions/input/gap-development_Alice_Kate_John_Michael.tsv"
    val_path1  = "../ensemble/model_7_submissions/input/gap-validation_Alice_Kate_John_Michael.tsv"

    dev_path2  = "../ensemble/model_7_submissions/input/gap-development_Elizabeth_Mary_James_Henry.tsv"
    val_path2  = "../ensemble/model_7_submissions/input/gap-validation_Elizabeth_Mary_James_Henry.tsv"

    dev_path3  = "../ensemble/model_7_submissions/input/gap-development_Kate_Elizabeth_Michael_James.tsv"
    val_path3  = "../ensemble/model_7_submissions/input/gap-validation_Kate_Elizabeth_Michael_James.tsv"

    dev_path4  = "../ensemble/model_7_submissions/input/gap-development_Mary_Alice_Henry_John.tsv"
    val_path4  = "../ensemble/model_7_submissions/input/gap-validation_Mary_Alice_Henry_John.tsv"





    ### da qui test val e dev path sono corretti come tu pensi che siano utilizzati




    logger.info ("building model ")
    model_squad_inst = model_squad ("model_9/weights_a1")
    model_squad_inst = model_squad ("model_9/weights_a2")
    model_squad_inst = model_squad ("model_9/weights_a3")
    model_squad_inst = model_squad ("model_9/weights_a4")





    logger.info ("training model ")
    model_squad_inst.train(dev_path1,val_path1)
    model_squad_inst.train(dev_path2,val_path2)
    model_squad_inst.train(dev_path3,val_path3)
    model_squad_inst.train(dev_path4,val_path4)
