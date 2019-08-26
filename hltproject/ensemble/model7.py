
import logging
import os

from common_interface import model
from model_7.Step1_preprocessing import original_notebook_preprocessing

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
        pass

        
    def train(self, train_set, validation_set, weight_folder_path ):
        os.makedirs(weight_folder_path+"/output", exist_ok=True)
        os.makedirs(weight_folder_path+"/input", exist_ok=True)
        os.makedirs(weight_folder_path+"/embeddings", exist_ok=True)
        logger.info ("preprocessing train set")
        original_notebook_preprocessing (False, weight_folder_path, train_set)
        
        logger.info ("preprocessing validation set")
        original_notebook_preprocessing (False, weight_folder_path, validation_set)
        

    def evaluate(self, val_df, weight_folder_path="model_7_weights" ):
        pass

#UNIT TESTS
if __name__ == "__main__":
    test_path = "../datasets/gap-light.tsv"
    dev_path = "../datasets/gap-light.tsv"
    val_path = "../datasets/gap-light.tsv"

    model7_instance = Model7 ()
    model7_instance.train ( dev_path, val_path, "model_7_weights")