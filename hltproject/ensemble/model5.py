
import logging
import os
import sys
from shutil import copyfile
from common_interface import model
import requests, zipfile, io
import subprocess

logger = logging.getLogger ( __name__ )

class Model5(model):

    def __init__(self):
        self.train_set, self.dev_set = '', ''
 
        
    def train(self, train_set, dev_set, weight_folder_path):
        bert_large_url = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
        
        if os.path.isdir('model_5/BERT/uncased_L-24_H-1024_A-16') == False:
            print("Downloading uncased_L-24_H-1024_A-16.zip")
            r = requests.get(bert_large_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("model_5/BERT/")
 
        # python3 gap_ken_gap_classifier.py --pre_train --use_tpu=false
        # flags.DEFINE_bool("pre_train", False, "Run pre-training to create TFRecords.")
        proc = subprocess.Popen([
            'python3',
            'model_5/gap_ken_gap_classifier.py',
            '--pre_train',
            '--use_tpu=false',
            '--train_data_path=' + train_set
            ],stdout=sys.stdout, stderr=sys.stderr).communicate()
        
        # python3 gap_ken_gap_classifier.py --do_train --use_tpu=false
        # flags.DEFINE_bool("do_train", False, "Whether to run training.")
        proc = subprocess.Popen([
            'python3',
            'model_5/gap_ken_gap_classifier.py',
            '--do_train',
            '--use_tpu=false',
            '--train_data_path=' + train_set
            ],stdout=sys.stdout, stderr=sys.stderr).communicate()
 
        # python3 gap_ken_gap_classifier.py --do_eval --use_tpu=false  
        # dev_data_path =       
        # flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
        proc = subprocess.Popen([
            'python3',
            'model_5/gap_ken_gap_classifier.py',
            '--do_eval',
            '--use_tpu=false',
            '--dev_data_path=' + dev_set
            ],stdout=sys.stdout, stderr=sys.stderr).communicate()

    def evaluate(self, val_set, weight_folder_path="model_5_weights"):

        # python3 gap_ken_gap_classifier.py --do_predict --use_tpu=false

        # flags.DEFINE_bool("do_predict", False, "Predict mode on the test set.")
        proc = subprocess.Popen([
            'python3',
            'model_5/gap_ken_gap_classifier.py',
            '--do_predict',
            '--use_tpu=false',
            '--test_data_path=' + val_set,
            '--output_dir=model_5/output/',
            '--output_file=output.csv'
            ],stdout=sys.stdout, stderr=sys.stderr).communicate()

        result = pd.read_csv("./model_5/data/output.csv", delimiter=",")
        return result.values
            
#RUN the model
if __name__ == "__main__":
    #TESTS with light dataset
    # test_path = "../datasets/gap-light.tsv"
    # dev_path = "../datasets/gap-light.tsv"
    # val_path = "../datasets/gap-light.tsv"

    test_path = "../datasets/gap-test.tsv"
    dev_path = "../datasets/gap-development.tsv"
    val_path = "../datasets/gap-validation.tsv"

    stage2_path = "../datasets/gap_validation_stage2.tsv"
    
    model5_instance = Model5()
    model5_instance.train ( test_path, dev_path, "model_7_weights")
    # model7_instance.evaluate (stage2_path, "model_7_weights" )
