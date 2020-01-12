
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

    
    test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
    dev_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv"
    val_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv"
    '''


    #per trainare e testare piu velocemente, sono solo 5 esempi
    test_path = "../datasets/gap-light.tsv"
    dev_path = "../datasets/gap-light.tsv"
    val_path = "../datasets/gap-light.tsv"
	'''


    val_examples_df = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")

    
    test_df_prod = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
    test_df_prod = test_df_prod.copy()
    test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]


    logger.info ("building model ")
    model_squad_inst = model_squad ("model_9/weights")
    model_swag_inst = model_swag ("model_9/weights")
    #model_SpanExtractor_inst = model_SpanExtractor ("model_9/weights")



    #logger.info ("training model ")
    #model_SpanExtractor_inst.train(dev_path,val_path)


    logger.info ("evaluating ")
    val_probas_no_i_squad = model_squad_inst.evaluate( val_examples_df )
    val_probas_no_i_swag = model_swag_inst.evaluate( val_examples_df )
    #val_probas_no_i_SpanExtractor = model_SpanExtractor_inst.evaluate( test_path ) #questo prende un path gli altri prendono un pd

    print(val_probas_no_i_swag)

    #exit()
    val_probas_df_squad= pd.DataFrame([test_df_prod.ID, val_probas_no_i_squad[:,0], val_probas_no_i_squad[:,1], val_probas_no_i_squad[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_swag= pd.DataFrame([test_df_prod.ID, val_probas_no_i_swag[:,0], val_probas_no_i_swag[:,1], val_probas_no_i_swag[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    #val_probas_df_SpanExtractor= pd.DataFrame([test_df_prod.ID, val_probas_no_i_SpanExtractor[:,0], val_probas_no_i_SpanExtractor[:,1], val_probas_no_i_SpanExtractor[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()

    val_probas_df_squad.to_csv('stage1_swag_only_my_w.csv', index=False)
    val_probas_df_swag.to_csv('stage1_swag_only_my_QA_w.csv', index=False)
    #val_probas_df_SpanExtractor.to_csv('stage1_swag_only_my_SEQ_w.csv', index=False)


    test_path = "../datasets/gap-test.tsv"


    print("loss squad")
    print(compute_loss("stage1_swag_only_my_w.csv",test_path))

    print("loss swag")
    print(compute_loss("stage1_swag_only_my_QA_w.csv",test_path))

    print("SEQ squad")
    #print(compute_loss("stage1_swag_only_my_SEQ_w.csv",test_path))

    #for i in range(len(y_test)):
    #    y_one_hot[i, y_test[i]] = 1
    #log_loss(y_one_hot, pred)


    #submission_df = pd.DataFrame([test_df_prod.ID, val_probas[:,0], val_probas[:,1], val_probas[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    #submission_df.to_csv('stage2_swag_only.csv', index=False)



'''

'''