from multi import *


class model(object):

    
    def train(self, train_set, vallidation_set, weight_folder_path):
        '''
         train the model on hte validation and traning set chosen, using a k-fold cross validation
         
         \param path of the train_set to be used
         \param path of the vallidation_set to be used
         \param weight_folder_path path to folder where after the training the model weight will be saved


        '''
        pass

    def evaluate(self, val_df, weight_folder_path):
        '''
         Load the saved model from the path chosen and  return and array witht the probability of each class for the set given in input
         
         \param path of the val_df set to be avaluated
         \param weight_folder_path path to folder where after the training the model weight will be saved


        '''
        pass

