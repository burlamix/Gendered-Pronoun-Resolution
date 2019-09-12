
import logging
import os
import pandas as pd
import numpy as np

from common_interface import model
from model_9.utils import BERTSpanExtractor
from sklearn.metrics import log_loss

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)


from hltproject.score.score import compute_loss_df

#from dataset_utils import compute_loss_simo

logger = logging.getLogger ( __name__ )





class model9(model):
    '''
        wrapper for 9th place model
        code: https://github.com/rakeshchada/corefqa
        paper: https://arxiv.org/pdf/1906.03695.pdf

    '''
    def __init__(self,weight_path):

        swag_runner = BERTSpanExtractor(None, None, None,  bert_model='bert-large-uncased')
        self.runner = swag_runner
        self.weight_path = weight_path

    def train(self, train_set, vallidation_set ):

        self.runner.train( train_set, vallidation_set, self.weight_path, n_splits=4)


    #forse qui sarebbe meglio riuscire a salvare i pvari pesi tutti nello stesso pickle 
#    def evaluate(self, val_df ):
    
#        return  self.runner.my_evaluate( val_df, self.weight_path, is_test=False)
    
    '''
    def fit(self, val_df , a ):

        return  self.runner.my_evaluate( val_df, self.weight_path, is_test=False)


    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"weight_path": self.weight_path}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    '''

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
    test_df_prod = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
    test_df_prod = test_df_prod.copy()
    test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]


    logger.info ("\n\nbuilding model ")
    model_9_inst = model9 ("model_9/weights")


    logger.info ("\n\ntraining model ")
    model_9_inst.train(dev_path,val_path)
    logger.info ("\n\n\n\ntraining finished ")


    logger.info ("evaluating ")
    val_probas_no_i = model_9_inst.evaluate( test_path )
    logger.info ("evaluating finished")

    print(val_probas_no_i)
    #test_path = "../datasets/gap-test.tsv"



    #val_probas = np.insert(val_probas_no_i, 0, np.arange(2000), axis=1)

    #print(val_probas)


    val_probas_df= pd.DataFrame([test_df_prod.ID, val_probas_no_i[:,0], val_probas_no_i[:,1], val_probas_no_i[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()

    val_probas_df.to_csv('stage1_swag_only_my_SEQ_w.csv', index=False)


    #print(compute_loss_df(val_probas_df,test_path))

    #for i in range(len(y_test)):
    #    y_one_hot[i, y_test[i]] = 1
    #log_loss(y_one_hot, pred)


    #submission_df = pd.DataFrame([test_df_prod.ID, val_probas[:,0], val_probas[:,1], val_probas[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    #submission_df.to_csv('stage2_swag_only.csv', index=False)



'''

'''