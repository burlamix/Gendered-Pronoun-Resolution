import logging
import os
import pandas as pd
import numpy as np

from model_9e import model_9e
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

    #test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
    test_path = "../datasets/gap-test.tsv"


    test_examples_df = pd.read_csv(test_path, delimiter="\t")

    test_df_prod = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
    test_df_prod = test_df_prod.copy()
    test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]
    ### da qui test val e dev path sono corretti come tu pensi che siano utilizzati




    logger.info ("building model ")
    model_9_inst1 = model_9e("model_9/weights_a1")
    model_9_inst2 = model_9e("model_9/weights_a2")
    model_9_inst3 = model_9e("model_9/weights_a3")
    model_9_inst4 = model_9e("model_9/weights_a4")




    logger.info ("training model ")
    model_9_inst1.train(dev_path1,val_path1)
    model_9_inst2.train(dev_path2,val_path2)
    model_9_inst3.train(dev_path3,val_path3)
    model_9_inst4.train(dev_path4,val_path4)


    model_e_inst = model_e([model_9_inst1,model_9_inst2,model_9_inst3,model_9_inst4])


    logger.info ("evaluating model ")

    res = model_9_inst1.evaluate(test_examples_df)
    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    print("loss 1 ")
    print(compute_loss("elim.csv",test_path))

    res = model_9_inst2.evaluate(test_examples_df)
    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    print("loss 2 ")
    print(compute_loss("elim.csv",test_path))

    res = model_9_inst3.evaluate(test_examples_df)
    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    print("loss 3 ")
    print(compute_loss("elim.csv",test_path))

    res = model_9_inst4.evaluate(test_examples_df)
    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    print("loss 4 ")
    print(compute_loss("elim.csv",test_path))

    res = model_e_inst.evaluate(test_examples_df,combination="mean")
    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    print("loss ensambled ")
    print(compute_loss("elim.csv",test_path))