
from common_interface import model
import random
import pandas as pd

class modelRand(model):
    ''' 
        model that always output random numbers
        only useful for testing
    '''
    
    def __init__(self, weights_path):
        random.seed (42)

    def train(self, train_fname, validation_fname ):
        pass

    def evaluate(self, test_set_fname ):
        test_df = pd.read_csv(test_set_fname, delimiter="\t")
        return  [[random.random(), random.random(), random.random()] for i in range (len(test_df))]
