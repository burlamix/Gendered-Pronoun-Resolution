from multi import *


class model(object):

    
    def train(self, train_set, vallidation_set, weight_folder_path):
        '''
         train the model on hte validation and traning set chosen, using a k-fold cross validation
         
         \param train_set to be used
         \param vallidation_set to be used
         \param weight_folder_path path to folder where after the training the model weight will be saved


        '''
        pass

    def evaluate(self, val_df, weight_folder_path):
        '''
         Load the saved model from the path chosen and  return and array witht the probability of each class for the set given in input
         
         \param val_df set to be avaluated
         \param weight_folder_path path to folder where after the training the model weight will be saved


        '''
        pass


class model_9(model):

    def __init__(self):

        #problema cerca di inizializzare l'oggetto senza dover istanziare i dataset, occupo memoria per nulla.
        swag_runner = BertSwagRunner(None, None, None, num_train_epochs=1, bert_model='bert-large-uncased')
        self.runner = swag_runner

    def train(self, train_set, vallidation_set, weight_folder_path ):

        self.runner.train( train_set, vallidation_set, weight_folder_path, n_splits=4)


    #forse qui sarebbe meglio riuscire a salvare i pvari pesi tutti nello stesso pickle 
    def evaluate(self, val_df, weight_folder_path="model_9" ):

        return  self.runner.my_evaluate( val_df, weight_folder_path, is_test=False)



'''
test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
dev_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv"
val_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv"
'''

#per trainare e testare più velocemente, sono solo 5 esempi
test_path = "../datasets/gap-light.tsv"
dev_path = "../datasets/gap-light.tsv"
val_path = "../datasets/gap-light.tsv"

dev_df = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
test_df = pd.read_csv(dev_path, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")
val_df = pd.read_csv(val_path, delimiter="\t")


print("\n\n\n\n         building model         \n\n")
model_9_inst = model_9()



print("\n\n\n\n         training model         \n\n")
model_9_inst.train(dev_df,val_df,"model_9")



print("\n\n\n\n         evaluating         \n\n")
val_probas = model_9_inst.evaluate( val_df,"model_9")

print("val_probas")
print(val_probas)


submission_df = pd.DataFrame([test_df_prod.ID, val_probas[:,0], val_probas[:,1], val_probas[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()

submission_df.to_csv('stage2_swag_only.csv', index=False)