
class Model7(model):
    '''
        wrapper for 7th place model
        code: https://github.com/boliu61/gendered-pronoun-resolution
        paper: https://arxiv.org/abs/1905.01780

    '''
        
    def train(self, train_set, dev_set, weight_folder_path ):
        pass
        

    def evaluate(self, val_set, weight_folder_path="model_7_weights" ):
        #read predictions from file ./model_7_submissions/sub_end2end_gap_validation_stage2.csv and output them
        pass


#RUN the model
if __name__ == "__main__":
    model_7_instance = Model7()
    
    predictions = model_7_instance.evaluate ()