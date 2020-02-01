
from common_interface import model
import pandas as pd

class modelFile(model):
    ''' 
        model that reads the predictions from a file and outputs them
        only useful for testing or for avoid models re-training
    '''
    
    def __init__(self, file_path):
        self._file_path = file_path

    def train(self, train_fname, validation_fname ):
        pass

    def evaluate(self, test_set_fname ):
        predictions = pd.read_csv(self._file_path, delimiter=",")
        return predictions.values[:,1:]

if __name__ == "__main__":
    
    dev_path = "../datasets/gap-light.tsv"
    val_path = "../datasets/gap-light.tsv"
    test_path = "../datasets/gap-light.tsv"
    
    mf = modelFile ("single_models_predictions/model5_anonymized_1_predictions.csv")
    mf.train ( dev_path, val_path )
    predictions = mf.evaluate ( test_path )

    print ("predictions\n{}".format (predictions))
