
from common_interface import model

class modelAllZeroThrees(model):
    ''' 
        model that always output 0.33333 for each label
        only useful for testing
    '''

    def __init__(self, weights_path):
        pass

    def train(self, train_fname, validation_fname ):
        pass

    def evaluate(self, test_df ):
        return  [[1/3, 1/3, 1/3] for i in range (len(test_df))]
