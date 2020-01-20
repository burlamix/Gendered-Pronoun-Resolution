
import logging
import os
import pickle
import math

from common_interface import model

from hltproject.score.score import compute_loss
import hltproject.utils.config as cutils

from modelRand import modelRand
from modelAllZeroThrees import modelAllZeroThrees

import pandas as pd
import numpy as np

logging.config.dictConfig(cutils.load_logger_config_file())
logger = logging.getLogger ( "model_e" )

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

    def __init__(self,modelli, model_names = None):
        self.modelli = modelli
        self.model_names = []
        
        self._OUTPUT_FOLDER = "predictions"

        if model_names is None:
            self.model_names = [ "model_number_{}".format(i) for i in range (len(modelli))]
        else:
            assert len(modelli) == len(model_names)
            self.model_names = model_names
        
        os.makedirs (self._OUTPUT_FOLDER, exist_ok=True)

    def train(self,train_set, validation_set):

        for modello in self.modelli:
            modello.train( train_set, validation_set )

    def evaluate(self,dataset,combination="mean"):

        risultati = []

        for modello, model_name in zip (self.modelli, self.model_names):
            logger.info ("evaluating {}".format(model_name))
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
        
    def evaluate_list(self,datasets_fnames,combination="mean",report_fname=None):

        assert len (datasets_fnames) == len (self.modelli)

        fout_report = None

        # if report_fname is not None writes each individual model performance and the ensemble performance on a file
        # and writes the prediction for each model on a CSV file
        if report_fname:
            report_fname = self._OUTPUT_FOLDER + "/" + report_fname
            fout_report = open (report_fname, "w")
            print ("Model name\tloss", file=fout_report)
            test_set = None
            test_path = None

        risultati = []
        
        for modello,test_set_fname,model_name in zip(self.modelli,datasets_fnames, self.
            ):
            logger.info ("evaluating {}".format(model_name))

            test_set = pd.read_csv(test_set_fname, delimiter="\t")

            res = np.asarray (modello.evaluate(test_set_fname))
            risultati.append( res )

            if fout_report:
                
                test_path = test_set_fname

                val_probas_df_e = pd.DataFrame([test_set.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
                prediction_fname = self._OUTPUT_FOLDER + "/" + "{}_predictions.csv".format(model_name)
                val_probas_df_e.to_csv(prediction_fname, index=False)
                loss = compute_loss(prediction_fname,test_set_fname, print=False)

                logger.info ("loss for model {}: {} - predictions written to {}".format (model_name, loss, prediction_fname))
                print ("{}\t{}".format(model_name, loss), file=fout_report)

        out = None
        if combination == "mean":
            out = np.mean(risultati, axis=0)
        elif combination == "max":
            out = np.asarray(max_voting(risultati))
        elif combination == "min":
            out = np.asarray(min_voting(risultati))
        elif combination == "min_entropy":
            out = np.asarray(min_entropy(risultati))
        else:
            out = np.mean(risultati, axis=0)
        
        # Computing ensemble performances
        if fout_report:
          
            val_probas_df_e = pd.DataFrame([test_set.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
            prediction_fname = self._OUTPUT_FOLDER + "/" + "ensemble_predictions.csv"
            val_probas_df_e.to_csv(prediction_fname, index=False)
            loss = compute_loss(prediction_fname,test_path, print=False)

            logger.info ("loss for ensemble: {} - predictions written to {}".format (loss, prediction_fname))
            print ("{}\t{}".format("ensemble", loss), file=fout_report)
            logger.info ("Done. Report written to {}".format(report_fname))

        return out


#UNIT TESTS
if __name__ == "__main__":


    test_path = "../datasets/gap-test.tsv"
    dev_path = "../datasets/gap-development.tsv"
    val_path = "../datasets/gap-validation.tsv"

    m1 = modelAllZeroThrees ("")
    m2 = modelAllZeroThrees ("")
    m3 = modelRand ("")
    
    modelli = [m1, m2, m3]
    model_names = ["ZT1", "ZT2", "Random"]

    logger.info ("building ensemble model ")
    model_e_inst = model_e(modelli, model_names)

    test_df_prod = pd.read_csv(test_path, delimiter="\t")
    test_df_prod = test_df_prod.copy()
    test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]


    logger.info ("evaluating model with the same test dataset for each model")
    res = model_e_inst.evaluate(test_df_prod,combination="mean")


    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    loss = compute_loss("elim.csv",test_path, print=False)
    logger.info ("ensemble loss  {}".format(loss))
    
    logger.info ("evaluating model with different test datasets for each model (no reporting)")
    res = model_e_inst.evaluate_list([test_path]*3)

    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    loss = compute_loss("elim.csv",test_path, print=False)
    logger.info ("ensemble loss  {}".format(loss))
    
    logger.info ("evaluating model with different test datasets for each model (with reporting)")
    res = model_e_inst.evaluate_list([test_path]*3, report_fname="report.tsv")

