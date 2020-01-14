
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
        

    def train(train_set, validation_set):

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
    test_examples_df = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")


    logger.info ("building model ")
    model_squad_inst1 = model_squad ("model_9/weights_a1")
    model_squad_inst2 = model_squad ("model_9/weights_a2")
    model_squad_inst3 = model_squad ("model_9/weights_a3")
    model_squad_inst4 = model_squad ("model_9/weights_a4")
    #model_squad_inst = model_squad ("model_9/weights")
    #model_swag_inst = model_swag ("model_9/weights")

    modelli = [model_squad_inst1,model_squad_inst2,model_squad_inst3,model_squad_inst4]



    logger.info ("building model ")
    model_e_inst = model_e(modelli)



    logger.info ("evaluating model ")
    res = model_e_inst.evaluate(test_examples_df,combination="mean")





    test_df_prod = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
    test_df_prod = test_df_prod.copy()
    test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]


    #val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()


    #val_probas_df_e.to_csv('stage1_ee_my_pred.csv', index=False)


    test_path = "../datasets/gap-test.tsv"

    #print("loss ensambled ")
    #print(compute_loss("stage1_ee_my_pred.csv",test_path))


    #print("loss squad")
    #print(compute_loss("stage1_swag_only_my_w.csv",test_path))

    #print("loss swag")
    #print(compute_loss("stage1_swag_only_my_QA_w.csv",test_path))



    #for fast testing
    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    print("loss  ")
    print(compute_loss("elim.csv",test_path))