


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




print("\n\n\n\n         building model         \n\n")
model_9_inst = model_9()



print("\n\n\n\n         training model         \n\n")
model_9_inst.train(dev_path,val_path,"model_9")



print("\n\n\n\n         evaluating         \n\n")
val_probas = model_9_inst.evaluate( test_path,"model_9")

print("val_probas")
print(val_probas)


submission_df = pd.DataFrame([test_df_prod.ID, val_probas[:,0], val_probas[:,1], val_probas[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()

submission_df.to_csv('stage2_swag_only.csv', index=False)