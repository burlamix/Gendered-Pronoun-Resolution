import logging
import os
import pandas as pd
import numpy as np

from model_9e import model_9e
from model_e import model_e
from model5 import Model5


import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)


logger = logging.getLogger ( __name__ )

def main_with_retraining ():

    dev_path1  = "../datasets/gap_development_stage2_Alice_Kate_John_Michael.tsv"
    val_path1  = "../datasets/gap_validation_stage2_Alice_Kate_John_Michael.tsv"
    test_path1 = "../datasets/test_stage_2_with_labels_Alice_Kate_John_Michael.tsv"

    dev_path2  = "../datasets/gap_development_stage2_Elizabeth_Mary_James_Henry.tsv"
    val_path2  = "../datasets/gap_validation_stage2_Elizabeth_Mary_James_Henry.tsv"
    test_path2 = "../datasets/test_stage_2_with_labels_Elizabeth_Mary_James_Henry.tsv"

    dev_path3  = "../datasets/gap_development_stage2_Kate_Elizabeth_Michael_James.tsv"
    val_path3  = "../datasets/gap_validation_stage2_Kate_Elizabeth_Michael_James.tsv"
    test_path3 = "../datasets/test_stage_2_with_labels_Kate_Elizabeth_Michael_James.tsv"

    dev_path4  = "../datasets/gap_development_stage2_Mary_Alice_Henry_John.tsv"
    val_path4  = "../datasets/gap_validation_stage2_Mary_Alice_Henry_John.tsv"
    test_path4 = "../datasets/test_stage_2_with_labels_Mary_Alice_Henry_John.tsv"

    dev_path   = "../datasets/gap_development_stage2.tsv"
    val_path   = "../datasets/gap_validation_stage2.tsv"
    test_path  = "../datasets/test_stage_2_with_labels.tsv"
    
    logger.info ("building models ")

    model9_original = model_9e("model_9/weights_o_stage2")
    model9_anonymized1 = model_9e("model_9/weights_a1_stage2")
    model9_anonymized2 = model_9e("model_9/weights_a2_stage2")
    model9_anonymized3 = model_9e("model_9/weights_a3_stage2")
    model9_anonymized4 = model_9e("model_9/weights_a4_stage2")

    model5_original  = Model5(weight_folder_path="model_5_o_stage2")
    model5_anonymized1 = Model5(weight_folder_path="model_5_a1_stage2")
    model5_anonymized2 = Model5(weight_folder_path="model_5_a2_stage2")
    model5_anonymized3 = Model5(weight_folder_path="model_5_a3_stage2")
    model5_anonymized4 = Model5(weight_folder_path="model_5_a4_stage2")


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
 

    combinations_to_test = ["mean", "smoothed_voting"]

    istance_name4 = ["model9_original","model9_anonymized1","model9_anonymized2","model9_anonymized3","model9_anonymized4","model5_original","model5_anonymized1","model5_anonymized2","model5_anonymized3","model5_anonymized4"]
    istance_obj4  = [ model9_original , model9_anonymized1 , model9_anonymized2 , model9_anonymized3 , model9_anonymized4 , model5_original , model5_anonymized1 , model5_anonymized2 , model5_anonymized3 , model5_anonymized4]
    model_95_all = model_e(istance_obj4, istance_name4, 'predictions_model95_all_stage2')


    logger.info ("  ------------------------------------ evaluating model 9+5 all  ------------------------------------")
    for comb in combinations_to_test:
        model_95_all.evaluate_list([test_path,test_path1,test_path2,test_path3,test_path4,test_path,test_path1,test_path2,test_path3,test_path4],
                                                                                                            combination=comb,report_fname="model_95_all_"+comb)


#RUN the ensemblers
if __name__ == "__main__":

    main_with_retraining ()
