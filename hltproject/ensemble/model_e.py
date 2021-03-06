
import logging
import os
import pickle
import math
import uuid
import datetime

from common_interface import model

from hltproject.score.score import compute_loss
from hltproject.score.score import compute_squared_loss
import hltproject.utils.config as cutils

from modelRand import modelRand
from modelAllZeroThrees import modelAllZeroThrees
from modelFile import modelFile

from collections import Counter

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

def voting (valutaion_arrays):
    
    ris= np.asarray(valutaion_arrays)
    new_ris= []
    number_of_models = ris.shape[0]
    number_of_classes = ris.shape[2]

    # logger.debug ("voting. ris Shape: {}".format(ris.shape))
    
    for sent_id in range(ris.shape[1]):     

        predictions = ris[:, sent_id, :]
        votes = np.argmax (predictions, axis=1)
        votes_counter = Counter (votes)

        ensembled_predictions = [ votes_counter[i]/number_of_models for i in range(number_of_classes) ]

        # logger.debug ("            sentence: {}".format(sent_id))
        # logger.debug ("            predictions: {}".format(predictions))
        # logger.debug ("            votes: {}".format(votes))
        # logger.debug ("            votes counter: {}".format(votes_counter))
        # logger.debug ("            ensembled predictions: {}".format(ensembled_predictions))
        # input()

        new_ris.append(ensembled_predictions)
    
    return new_ris

def smoothed_voting (valutaion_arrays, smooth_by=0.01):
    
    ris= np.asarray(valutaion_arrays)
    new_ris= []
    number_of_models = ris.shape[0]
    number_of_classes = ris.shape[2]

    # assigned probability is smooth_by + p*( 1 - number_of_classes * smooth_by )
    # where p is the probability assigned by the "voting" method
    split = 1 - smooth_by * number_of_classes

    # logger.debug ("voting. ris Shape: {}".format(ris.shape))
    
    for sent_id in range(ris.shape[1]):     

        predictions = ris[:, sent_id, :]
        votes = np.argmax (predictions, axis=1)
        votes_counter = Counter (votes)

        ensembled_predictions = [ smooth_by + (votes_counter[i]/number_of_models)*split for i in range(number_of_classes) ]

        # logger.debug ("            sentence: {}".format(sent_id))
        # logger.debug ("            predictions: {}".format(predictions))
        # logger.debug ("            votes: {}".format(votes))
        # logger.debug ("            votes counter: {}".format(votes_counter))
        # logger.debug ("            ensembled predictions: {}".format(ensembled_predictions))
        # input()

        new_ris.append(ensembled_predictions)
    
    return new_ris

def max_simone(valutaion_arrays):

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
        min_entropy_so_far = math.inf
        for model_eval in ris[:, xid, :]:

            tot=0
            for x in model_eval:
                tot = tot + (x*math.log2(x))

            tot = -tot

            if(tot<min_entropy_so_far):
                a = model_eval
                min_entropy_so_far = tot

        new_ris.append(a)

    return new_ris


class model_e(model):

    def __init__(self,modelli, model_names = None, base_output_folder = "predictions"):
        self.modelli = modelli
        self.model_names = []
        
        self.base_output_folder = base_output_folder
        self.ensembler_id = uuid.uuid4 ()

        if model_names is None:
            self.model_names = [ "model_number_{}".format(i) for i in range (len(modelli))]
        else:
            assert len(modelli) == len(model_names)
            self.model_names = model_names        

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

        elif combination == "simone":
            return np.asarray(max_simone(risultati))

        elif combination == "min":
            return np.asarray(min_voting(risultati))

        elif combination == "min_entropy":
            return np.asarray(min_entropy(risultati))
        
        elif combination == "voting":
            return np.asarray(voting(risultati))
        
        elif combination == "smoothed_voting":
            return np.asarray(smoothed_voting(risultati))

        raise ValueError ("Wrong combination name {}".format(combination))
        
    def evaluate_list(self,datasets_fnames,combination="mean",report_fname=None):

        assert len (datasets_fnames) == len (self.modelli)
        
        fout_report = None
        output_folder = None

        # if report_fname is not None writes each individual model performance and the ensemble performance on a file
        # and writes the prediction for each model on a CSV file
        if report_fname:

            date = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

            output_folder = "{}_{}_{}_{}".format (self.base_output_folder, date, combination, self.ensembler_id)
            os.makedirs (output_folder, exist_ok=True)
            report_fname = output_folder + "/" + report_fname
            fout_report = open (report_fname, "w")
            print ("Model name\tlogloss\tsquared loss", file=fout_report)
            test_set = None
            test_path = None

        risultati = []
        
        for modello,test_set_fname,model_name in zip(self.modelli,datasets_fnames, self.model_names):
            logger.info ("evaluating {}".format(model_name))

            test_set = pd.read_csv(test_set_fname, delimiter="\t")

            res = np.asarray (modello.evaluate(test_set_fname))
            risultati.append( res )

            if fout_report:
                
                test_path = test_set_fname

                val_probas_df_e = pd.DataFrame([test_set.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
                prediction_fname = output_folder + "/" + "{}_predictions.csv".format(model_name)
                val_probas_df_e.to_csv(prediction_fname, index=False)
                
                logloss = compute_loss (prediction_fname,test_set_fname, enable_print=False)
                squaredloss = compute_squared_loss (prediction_fname,test_set_fname, enable_print=False)

                logger.info ("logloss      for model {}: {}".format (model_name, logloss))
                logger.info ("squared loss for model {}: {}".format (model_name, squaredloss))
                
                print ("{}\t{}\t{}".format(model_name, logloss, squaredloss), file=fout_report)

                
        out = None
        if combination == "mean":
            out = np.mean(risultati, axis=0)
        elif combination == "simone":
            out = np.asarray(max_simone(risultati))
        elif combination == "min":
            out = np.asarray(min_voting(risultati))
        elif combination == "min_entropy":
            out = np.asarray(min_entropy(risultati))
        elif combination == "voting":
            out = np.asarray(voting(risultati))
        elif combination == "smoothed_voting":
            out = np.asarray(smoothed_voting(risultati))
        else:
            raise ValueError ("Wrong combination name {}".format(combination))
        
        # Computing ensemble performances
        if fout_report:
          
            val_probas_df_e = pd.DataFrame([test_set.ID, out[:,0], out[:,1], out[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
            prediction_fname = output_folder + "/" + "ensemble_predictions.csv"
            val_probas_df_e.to_csv(prediction_fname, index=False)
            
            logloss = compute_loss(prediction_fname,test_path, enable_print=False)
            squaredloss = compute_squared_loss (prediction_fname,test_path, enable_print=False)

            logger.info ("logloss     for ensemble({}): {}".format (combination, logloss))
            logger.info ("squaredloss for ensemble({}): {}".format (combination, squaredloss))
            print ("ensemble({})\t{}\t{}".format(combination, logloss, squaredloss), file=fout_report)
            
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
    m4 = modelFile ("single_models_predictions/model5_anonymized_1_predictions.csv")
    
    modelli = [m1, m2, m3, m4]
    model_names = ["ZT1", "ZT2", "Random1", "File"]

    logger.info ("building ensemble model ")
    model_e_inst = model_e(modelli, model_names, 'test_ensemble')
    model_e_inst2 = model_e(modelli, model_names, 'test_ensemble')

    test_df_prod = pd.read_csv(test_path, delimiter="\t")
    test_df_prod = test_df_prod.copy()
    test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]


    logger.info ("evaluating model with the same test dataset for each model")
    res = model_e_inst.evaluate(test_path,combination="mean")

    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    loss = compute_loss("elim.csv",test_path, enable_print=False)
    logger.info ("ensemble loss  {}".format(loss))
    
    logger.info ("evaluating model with different test datasets for each model (no reporting)")
    res = model_e_inst.evaluate_list([test_path]*4)

    val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
    val_probas_df_e.to_csv('elim.csv', index=False)
    loss = compute_loss("elim.csv",test_path, enable_print=False)
    logger.info ("ensemble loss  {}".format(loss))
    
    logger.info ("evaluating model with different test datasets for each model (with reporting)")
    res = model_e_inst.evaluate_list([test_path]*4, report_fname="report.tsv")
    
    logger.info ("evaluating model another time (with reporting) - simone. Predictions should be saved in a different folder")
    res = model_e_inst.evaluate_list([test_path]*4, combination="simone", report_fname="report.tsv")
    
    logger.info ("evaluating model with another ensembler. Predictions should be saved in a different folder")
    res = model_e_inst2.evaluate_list([test_path]*4, combination="simone", report_fname="report.tsv")
    
    logger.info ("evaluating model with combination=voting. Predictions should be saved in a different folder")
    res = model_e_inst2.evaluate_list([test_path]*4, combination="voting", report_fname="report.tsv")
    
    logger.info ("evaluating model with combination=smoothed_voting. Predictions should be saved in a different folder")
    res = model_e_inst2.evaluate_list([test_path]*4, combination="smoothed_voting", report_fname="report.tsv")

    logger.info ("evaluating model with combination=min_entropy. Predictions should be saved in a different folder")
    res = model_e_inst2.evaluate_list([test_path]*4, combination="min_entropy", report_fname="report.tsv")

