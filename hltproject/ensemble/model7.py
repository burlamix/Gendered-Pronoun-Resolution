
import logging
import os
from shutil import copyfile

from common_interface import model
from model_7.Step1_preprocessing import original_notebook_preprocessing
from model_7.Step2_end2end_model import original_notebook_e2e

logger = logging.getLogger ( __name__ )

class Model7(model):
    '''
        wrapper for 7th place model
        code: https://github.com/boliu61/gendered-pronoun-resolution
        paper: https://arxiv.org/abs/1905.01780

        FIXME 2019-08-24
        This class requires the en_core_web_lg model for the spacy library to be downloaded.
        Apparently there is no way to download it internally from spacy.
        Please run the command
           python -m spacy download --user en_core_web_lg
    '''

    def __init__(self):
        self.train_set, self.dev_set = '', ''
 
        
    def train(self, train_set, dev_set, weight_folder_path ):

        self.train_set, self.dev_set = train_set, dev_set

        os.makedirs(weight_folder_path+"/output", exist_ok=True)
        os.makedirs(weight_folder_path+"/input", exist_ok=True)
        os.makedirs(weight_folder_path+"/embeddings", exist_ok=True)
        os.makedirs(weight_folder_path+"/sub", exist_ok=True)

        logger.info ("preprocessing train set")
        # original_notebook_preprocessing (False, weight_folder_path, train_set)
        
        logger.info ("preprocessing dev set")
        # original_notebook_preprocessing (False, weight_folder_path, dev_set)
        

    def evaluate(self, val_set, weight_folder_path="model_7_weights" ):

        if not self.train_set or not self.dev_set:
            raise RuntimeError ("model7: call train() before evaluate()")

        #original_notebook_preprocessing (False, weight_folder_path, val_set)
        
        copyfile ( "model_7/gap-development-corrected-74.tsv", weight_folder_path + "/input/" + 'gap-development-corrected-74.tsv' )
        copyfile ( "model_7/gap-test-val-85.tsv", weight_folder_path + "/input/" + 'gap-test-val-85.tsv' )
        copyfile ( "model_7/sample_submission_stage_1.csv", weight_folder_path + "/input/" + 'sample_submission_stage_1.csv' )

        for all_train in [ False, True ]:
            for CASED in [ True, False ]:
                original_notebook_e2e ( all_train, CASED, weight_folder_path, self.train_set, self.dev_set, val_set )

#RUN the model
if __name__ == "__main__":
    #TESTS with light dataset
    # test_path = "../datasets/gap-light.tsv"
    # dev_path = "../datasets/gap-light.tsv"
    # val_path = "../datasets/gap-light.tsv"

    test_path = "../datasets/gap-test.tsv"
    dev_path = "../datasets/gap-development.tsv"
    val_path = "../datasets/gap-validation.tsv"
    
    model7_instance = Model7 ()
    model7_instance.train ( test_path, dev_path, "model_7_weights")
    model7_instance.evaluate (val_path, "model_7_weights" )