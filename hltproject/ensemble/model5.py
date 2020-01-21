
import logging
import os
import sys
from shutil import copyfile
from common_interface import model
import requests, zipfile, io
import subprocess
import pandas as pd

logger = logging.getLogger ( __name__ )

class Model5(model):

    def __init__(self, weight_folder_path):
        self.weight_folder_path = weight_folder_path 
        
    def train(self, dev_set, val_set):
        bert_large_url = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"

        if os.path.isdir('model_5/bert/uncased_L-24_H-1024_A-16') == False:
            print("Downloading uncased_L-24_H-1024_A-16.zip")
            r = requests.get(bert_large_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("model_5/bert/")

        # python3 gap_ken_gap_classifier.py --pre_train --use_tpu=false
        # flags.DEFINE_bool("pre_train", False, "Run pre-training to create TFRecords.")
        print("**** PRE TRAIN ****")
        proc = subprocess.Popen([
            'python',
            'model_5/gap_ken_gap_classifier.py',
            '--output_dir=' + self.weight_folder_path,
            '--pre_train',
            '--use_tpu=false',
            '--train_data_path=' + dev_set
            ],stdout=sys.stdout, stderr=sys.stderr).communicate()

        # python3 gap_ken_gap_classifier.py --do_train --use_tpu=false
        # flags.DEFINE_bool("do_train", False, "Whether to run training.")
        print("**** TRAIN ****")
        proc = subprocess.Popen([
            'python',
            'model_5/gap_ken_gap_classifier.py',
            '--output_dir=' + self.weight_folder_path,
            '--do_train',
            '--use_tpu=false',
            '--train_data_path=' + dev_set
            ],stdout=sys.stdout, stderr=sys.stderr).communicate()

        # python3 gap_ken_gap_classifier.py --do_eval --use_tpu=false
        # dev_data_path =
        # flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
        print("**** EVAL ****")
        proc = subprocess.Popen([
            'python',
            'model_5/gap_ken_gap_classifier.py',
            '--output_dir=' + self.weight_folder_path,
            '--do_eval',
            '--use_tpu=false',
            '--dev_data_path=' + val_set
            ],stdout=sys.stdout, stderr=sys.stderr).communicate()

    def evaluate(self, test_set):

        # python3 gap_ken_gap_classifier.py --do_predict --use_tpu=false

        # flags.DEFINE_bool("do_predict", False, "Predict mode on the test set.")
        proc = subprocess.Popen([
            'python',
            'model_5/gap_ken_gap_classifier.py',
            '--output_dir=' + self.weight_folder_path,
            '--do_predict',
            '--use_tpu=false',
            '--test_data_path=' + test_set,
            '--output_file=output.csv'
            ],stdout=sys.stdout, stderr=sys.stderr).communicate()

        result = pd.read_csv(self.weight_folder_path + "/output.csv", delimiter=",")

        return result.values[:,1:]

# RUN the model
if __name__ == "__main__":
    
    dev_path = "../datasets/gap-light.tsv"
    val_path = "../datasets/gap-light.tsv"
    test_path = "../datasets/gap-light.tsv"
    
    model5_instance = Model5( "model_5_weights" )
    model5_instance.train ( dev_path, val_path )
    model5_instance.evaluate (test_path )
