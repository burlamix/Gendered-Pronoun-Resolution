
import logging
import os

from common_interface import model
from model_9.utils import *

logger = logging.getLogger ( __name__ )


class model9(model):
    '''
        wrapper for 9th place model
        code: https://github.com/rakeshchada/corefqa
        paper: https://arxiv.org/pdf/1906.03695.pdf

    '''
    def __init__(self,weight_path):

        swag_runner = BertSwagRunner(None, None, None, num_train_epochs=1, bert_model='bert-large-uncased')
        self.runner = swag_runner
        self.weight_path = weight_path

    def train(self, train_set, vallidation_set ):

        self.runner.train( train_set, vallidation_set, self.weight_path, n_splits=4)


    #forse qui sarebbe meglio riuscire a salvare i pvari pesi tutti nello stesso pickle 
    def evaluate(self, val_df ):

        return  self.runner.my_evaluate( val_df, self.weight_path, is_test=False)

    def fit(self, val_df , a ):

        return  self.runner.my_evaluate( val_df, self.weight_path, is_test=False)


    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"weight_path": self.weight_path}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


#UNIT TESTS
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

    test_df_prod = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
    test_df_prod = test_df_prod.copy()
    test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]


    logger.info ("building model ")
    model_9_inst = model9 ("model_9/weights")


    logger.info ("training model ")
    model_9_inst.train(dev_path,val_path)


    logger.info ("evaluating ")
    val_probas = model_9_inst.evaluate( test_path )


    print("val_probas")
    print(val_probas)


    #submission_df = pd.DataFrame([test_df_prod.ID, val_probas[:,0], val_probas[:,1], val_probas[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    #submission_df.to_csv('stage2_swag_only.csv', index=False)
