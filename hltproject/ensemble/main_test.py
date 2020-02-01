import logging
import os
import pandas as pd
import numpy as np

from model_9e import model_9e
from model_e import model_e
from model5 import Model5

from modelFile import modelFile


import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)


from hltproject.score.score import compute_loss_df
from hltproject.score.score import compute_loss

#from dataset_utils import compute_loss_simo

logger = logging.getLogger ( __name__ )

def main_with_retraining ():

    dev_path1  = "../ensemble/model_7_submissions/input/gap-development_Alice_Kate_John_Michael.tsv"
    val_path1  = "../ensemble/model_7_submissions/input/gap-validation_Alice_Kate_John_Michael.tsv"
    test_path1 = "../ensemble/model_7_submissions/input/gap-test_Alice_Kate_John_Michael.tsv"

    dev_path2  = "../ensemble/model_7_submissions/input/gap-development_Elizabeth_Mary_James_Henry.tsv"
    val_path2  = "../ensemble/model_7_submissions/input/gap-validation_Elizabeth_Mary_James_Henry.tsv"
    test_path2 = "../ensemble/model_7_submissions/input/gap-test_Elizabeth_Mary_James_Henry.tsv"

    test_path3 = "../ensemble/model_7_submissions/input/gap-test_Kate_Elizabeth_Michael_James.tsv"
    dev_path3  = "../ensemble/model_7_submissions/input/gap-development_Kate_Elizabeth_Michael_James.tsv"
    val_path3  = "../ensemble/model_7_submissions/input/gap-validation_Kate_Elizabeth_Michael_James.tsv"

    dev_path4  = "../ensemble/model_7_submissions/input/gap-development_Mary_Alice_Henry_John.tsv"
    val_path4  = "../ensemble/model_7_submissions/input/gap-validation_Mary_Alice_Henry_John.tsv"
    test_path4 = "../ensemble/model_7_submissions/input/gap-test_Mary_Alice_Henry_John.tsv"

    val_path   = "../datasets/gap-validation.tsv"
    test_path  = "../datasets/gap-test.tsv"
    dev_path   = "../datasets/gap-development.tsv"

    logger.info ("building models ")

    model9_original = model_9e("model_9/weights_c_f")
    model9_anonymized1 = model_9e("model_9/weights_a1_f")
    model9_anonymized2 = model_9e("model_9/weights_a2_f")
    model9_anonymized3 = model_9e("model_9/weights_a3_f")
    model9_anonymized4 = model_9e("model_9/weights_a4_f")

    model5_original  = Model5(weight_folder_path="model_5_c_f")
    model5_anonymized1 = Model5(weight_folder_path="model_5_a1_f")
    model5_anonymized2 = Model5(weight_folder_path="model_5_a2_f")
    model5_anonymized3 = Model5(weight_folder_path="model_5_a3_f")
    model5_anonymized4 = Model5(weight_folder_path="model_5_a4_f")

    logger.info ("training model 9")
    model9_original.train(dev_path, val_path)
    model9_anonymized1.train(dev_path1, val_path1)
    model9_anonymized2.train(dev_path2, val_path2)
    model9_anonymized3.train(dev_path3, val_path3)
    model9_anonymized4.train(dev_path4, val_path4)

    logger.info ("training model 5")
    model5_original.train(dev_path, val_path)
    model5_anonymized1.train(dev_path1, val_path1)
    model5_anonymized2.train(dev_path2, val_path2)
    model5_anonymized3.train(dev_path3, val_path3)
    model5_anonymized4.train(dev_path4, val_path4)
 

    combinations_to_test = ["mean", "min_entropy", "voting", "smoothed_voting"]

    istance_name1= ["model9_original","model5_original"]
    istance_obj1  = [model9_original,model5_original]
    model_95 = model_e(istance_obj1, istance_name1, 'predictions_model95')

    istance_name2 = ["model9_original","model9_anonymized1","model9_anonymized2","model9_anonymized3","model9_anonymized4"]
    istance_obj2  = [ model9_original , model9_anonymized1 , model9_anonymized2 , model9_anonymized3 , model9_anonymized4]
    model_9_all = model_e(istance_obj2, istance_name2, 'predictions_model9all')

    istance_name3 =["model5_original","model5_anonymized1","model5_anonymized2","model5_anonymized3","model5_anonymized4"]
    istance_obj3  =[ model5_original , model5_anonymized1 , model5_anonymized2 , model5_anonymized3 , model5_anonymized4]
    model_5_all = model_e(istance_obj3, istance_name3, 'predictions_model5all')

    istance_name4 = ["model9_original","model9_anonymized1","model9_anonymized2","model9_anonymized3","model9_anonymized4","model5_original","model5_anonymized1","model5_anonymized2","model5_anonymized3","model5_anonymized4"]
    istance_obj4  = [ model9_original , model9_anonymized1 , model9_anonymized2 , model9_anonymized3 , model9_anonymized4 , model5_original , model5_anonymized1 , model5_anonymized2 , model5_anonymized3 , model5_anonymized4]
    model_95_all = model_e(istance_obj4, istance_name4, 'predictions_model95_all')

    logger.info ("  ------------------------------------ evaluating model 5 all  ------------------------------------")
    for comb in combinations_to_test:
        model_5_all.evaluate_list([test_path,test_path1,test_path2,test_path3,test_path4],combination=comb,report_fname="model_5_all_"+comb)

    logger.info ("  ------------------------------------ evaluating model 9 all  ------------------------------------")
    for comb in combinations_to_test:
        model_9_all.evaluate_list([test_path,test_path1,test_path2,test_path3,test_path4],combination=comb,report_fname="model_9_all_"+comb)

    logger.info ("  \n\n\n\n ------------------------------------ evaluating model 9+5  ------------------------------------")
    for comb in combinations_to_test:
        model_95.evaluate_list([test_path,test_path],combination=comb,report_fname="model_95_"+comb)

    logger.info ("  ------------------------------------ evaluating model 9+5 all  ------------------------------------")
    for comb in combinations_to_test:
        model_95_all.evaluate_list([test_path,test_path1,test_path2,test_path3,test_path4,test_path,test_path1,test_path2,test_path3,test_path4],
                                                                                                            combination=comb,report_fname="model_95_all_"+comb)


def main_without_retraining ():

    test_path  = "../datasets/gap-test.tsv"

    model9_original = modelFile ("single_models_predictions/model9_original_predictions.csv")
    model9_anonymized1 = modelFile ("single_models_predictions/model9_anonymized_1_predictions.csv")
    model9_anonymized2 = modelFile ("single_models_predictions/model9_anonymized_2_predictions.csv")
    model9_anonymized3 = modelFile ("single_models_predictions/model9_anonymized_3_predictions.csv")
    model9_anonymized4 = modelFile ("single_models_predictions/model9_anonymized_4_predictions.csv")
    
    model5_original  = modelFile ("single_models_predictions/model5_original_predictions.csv")
    model5_anonymized1 = modelFile ("single_models_predictions/model5_anonymized_1_predictions.csv")
    model5_anonymized2 = modelFile ("single_models_predictions/model5_anonymized_2_predictions.csv")
    model5_anonymized3 = modelFile ("single_models_predictions/model5_anonymized_3_predictions.csv")
    model5_anonymized4 = modelFile ("single_models_predictions/model5_anonymized_4_predictions.csv") 

    combinations_to_test = ["mean", "min_entropy", "voting", "smoothed_voting"]

    istance_name1= ["model9_original","model5_original"]
    istance_obj1  = [model9_original,model5_original]
    model_95 = model_e(istance_obj1, istance_name1, 'predictions_model95')

    istance_name2 = ["model9_original","model9_anonymized1","model9_anonymized2","model9_anonymized3","model9_anonymized4"]
    istance_obj2  = [ model9_original , model9_anonymized1 , model9_anonymized2 , model9_anonymized3 , model9_anonymized4]
    model_9_all = model_e(istance_obj2, istance_name2, 'predictions_model9all')

    istance_name3 =["model5_original","model5_anonymized1","model5_anonymized2","model5_anonymized3","model5_anonymized4"]
    istance_obj3  =[ model5_original , model5_anonymized1 , model5_anonymized2 , model5_anonymized3 , model5_anonymized4]
    model_5_all = model_e(istance_obj3, istance_name3, 'predictions_model5all')

    istance_name4 = ["model9_original","model9_anonymized1","model9_anonymized2","model9_anonymized3","model9_anonymized4","model5_original","model5_anonymized1","model5_anonymized2","model5_anonymized3","model5_anonymized4"]
    istance_obj4  = [ model9_original , model9_anonymized1 , model9_anonymized2 , model9_anonymized3 , model9_anonymized4 , model5_original , model5_anonymized1 , model5_anonymized2 , model5_anonymized3 , model5_anonymized4]
    model_95_all = model_e(istance_obj4, istance_name4, 'predictions_model95_all')

    logger.info ("  ------------------------------------ evaluating model 5 all  ------------------------------------")
    for comb in combinations_to_test:
        model_5_all.evaluate_list([test_path]*5,combination=comb,report_fname="model_5_all_"+comb)

    logger.info ("  ------------------------------------ evaluating model 9 all  ------------------------------------")
    for comb in combinations_to_test:
        model_9_all.evaluate_list([test_path]*5,combination=comb,report_fname="model_9_all_"+comb)

    logger.info ("  \n\n\n\n ------------------------------------ evaluating model 9+5  ------------------------------------")
    for comb in combinations_to_test:
        model_95.evaluate_list([test_path,test_path],combination=comb,report_fname="model_95_"+comb)

    logger.info ("  ------------------------------------ evaluating model 9+5 all  ------------------------------------")
    for comb in combinations_to_test:
        model_95_all.evaluate_list([test_path]*10,combination=comb,report_fname="model_95_all_"+comb)


#RUN the ensemblers
if __name__ == "__main__":

    # main_with_retraining ()
    main_without_retraining ()
