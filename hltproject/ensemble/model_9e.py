
import logging
import os
import pandas as pd
import numpy as np

from common_interface import model
from model_e import model_e
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


class model_9e(object):

    def __init__(self,weight_path):

        model_squad_inst = model_squad (weight_path+"_squad")
        model_swag_inst = model_swag (weight_path+"_swag")
        self.model_e = model_e([ model_squad_inst, model_swag_inst ])

    def train(self, train_set, vallidation_set ):

        self.model_e.train( train_set, vallidation_set)


    def evaluate(self, val_df ):
        
        val_df_df = pd.read_csv(val_df, delimiter="\t")

        return  self.model_e.evaluate( val_df_df )



#UNIT TESTS
if __name__ == "__main__":

    test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
    dev_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv"
    val_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv"




    logger.info ("building model ")
    model_e_inst = model_9e("model_9/weights_9e")


    #logger.info ("training model ")
    #model_e_inst.train(dev_path,val_path)


    logger.info ("evaluating model ")
    test_examples_df = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")
    res = model_e_inst.evaluate(test_path)





    test_df_prod = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
    test_df_prod = test_df_prod.copy()
    test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]

    test_path = "../datasets/gap-test.tsv"

    #for fast testing
    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    print("loss  ")
    print(compute_loss("elim.csv",test_path))