all_train = True  

import logging
import logging.config
import hltproject.utils.config as cutils

from shutil import copyfile

logging.config.dictConfig(
    cutils.load_logger_config_file())
logger = logging.getLogger ( __name__ )
logger.setLevel (logging.INFO)

def original_notebook_inference ( path, input_fname ):

  import numpy as np # linear algebra
  import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

  import os
  import zipfile
  import sys
  import datetime
  from glob import glob
  import gc
  from tqdm import tqdm
  import shutil
  import re

  path = path + '/'
  input_tsv = path+'input/'+os.path.basename (input_fname)

  logger.info ("original_notebook_inference input_tsv: {}".format(input_fname))

  use_lingui_features = True if all_train else False

  def create_input(embed_df, dist_df):
      
      if len(embed_df) != len(dist_df): print(len(embed_df), len(dist_df))
      assert len(embed_df) == len(dist_df)
      all_P, all_A, all_B = [] ,[] ,[]
      all_label = []
      all_dist_PA, all_dist_PB = [], []    
      
      for i in range(len(embed_df)):
          
          all_P.append(embed_df.loc[i, "emb_P"])
          all_A.append(embed_df.loc[i, "emb_A"])
          all_B.append(embed_df.loc[i, "emb_B"])
          all_dist_PA.append(dist_df.loc[i, "D_PA"])
          all_dist_PB.append(dist_df.loc[i, "D_PB"])                

      result_lst = [np.asarray(all_A), np.asarray(all_B), np.asarray(all_P),
                    np.expand_dims(np.asarray(all_dist_PA),axis=1),
                    np.expand_dims(np.asarray(all_dist_PB),axis=1)]

      if use_lingui_features:
        for col in dist_df.columns[2:].values:
          result_lst.append(np.expand_dims(dist_df[col].values,axis=1))
              
      return result_lst, all_label 

  # load previously extracted stage2 features

  LARGE = True

  def load_stage2_features(CASED):

    layer = "-4"
    
    suffix = layer
    if CASED: suffix += '_CASED'
    if LARGE: suffix += '_LARGE'

    json_suffix = '_fix_long_text.json'

    TTA_suffixes = [ \
                    '_Alice_Kate_John_Michael',
                    '_Elizabeth_Mary_James_Henry',
                    '_Kate_Elizabeth_Michael_James',
                    '_Mary_Alice_Henry_John']

    d_X_test = {}     # dict for test features

    for TTA_suffix in [''] + TTA_suffixes:
      gc.collect()

      num_test = pd.read_csv(input_tsv,sep='\t').shape[0]
      n_chunk = int(np.ceil(num_test/1000))
      for i in range(n_chunk):
        df_name = path+'output/contextual_embeddings_'+ os.path.basename(input_tsv).split('.')[0] + '_' + suffix +TTA_suffix+ f'_{i}'+ json_suffix
        logger.info ("loading stage2 dataframe {}".format ( df_name ))
        df_stage2_chunk =  pd.read_json(df_name).sort_index()  
        if i==0: df_stage2 = df_stage2_chunk.copy()
        else: df_stage2 = pd.concat([df_stage2,df_stage2_chunk])
        
      stage2_emb0 = df_stage2.reset_index(drop=True).copy()  

      stage2_dist_df = pd.read_csv(path+'output/stage2_dist_df.csv').reset_index(drop=True).copy()
      if use_lingui_features:
        stage2_lingui_df = pd.read_csv(path+'output/stage2_lingui_df.csv')
        stage2_dist_df = pd.concat([pd.read_csv(path+'output/stage2_dist_df.csv')[['D_PA','D_PB']], stage2_lingui_df], axis=1)

      # put into dictionary
      key = 'orig' if TTA_suffix=='' else TTA_suffix.strip('_')    

      X_test0, _ = create_input(stage2_emb0, stage2_dist_df)    
      d_X_test[key] = X_test0.copy()

    return d_X_test

  # from keras.layers import *
  import keras.backend as K
  # from keras.models import *
  from keras.layers import Input, Dense, Embedding, Activation, Dropout, Flatten, Multiply, Concatenate, Lambda
  from keras.models import Model, Sequential
  import keras
  from keras import optimizers
  from keras import callbacks
  from IPython.display import SVG
  from keras.utils.vis_utils import model_to_dot

  class End2End_NCR():
      
      def __init__(self, word_input_shape, dist_shape, embed_dim=20): 
          
          self.word_input_shape = word_input_shape
          self.dist_shape   = dist_shape
          self.embed_dim    = embed_dim
          self.buckets      = [1, 2, 3, 4, 5, 8, 16, 32, 64] 
          self.hidden_dim   = 150
          self.dense_layer_sizes = [512,32]
          self.dropout_rate = 0.6
          
      def build(self):
          
          A, B, P = Input((self.word_input_shape,)), Input((self.word_input_shape,)), Input((self.word_input_shape,))
          inputs = [A, B, P]
          if use_lingui_features: 
            num_lingui_features = pd.read_csv(path+'output/stage2_lingui_df.csv').shape[1]
            dist_inputs = [Input((1,)) for i in range(num_lingui_features+2)]
          else: 
            dist1, dist2 = Input((self.dist_shape,)), Input((self.dist_shape,))
            dist_inputs = [dist1, dist2]
          
          self.dist_embed = Embedding(10, self.embed_dim)
          self.ffnn       = Sequential([Dense(self.hidden_dim, use_bias=True),
                                      Activation('relu'),
                                      Dropout(rate=0.2, seed = 7),
                                      Dense(1, activation='linear')])              
          
          dist_embeds = [self.dist_embed(dist) for dist in dist_inputs[:2]]
          dist_embeds = [Flatten()(dist_embed) for dist_embed in dist_embeds]
          
          #Scoring layer
          #In https://www.aclweb.org/anthology/D17-1018, 
          #used feed forward network which measures if it is an entity mention using a score
          #because we already know the word is mention.
          #In here, I just focus on the pairwise score
          PA = Multiply()([inputs[0], inputs[2]])
          PB = Multiply()([inputs[1], inputs[2]])
          #PairScore: sa(i,j) =wa·FFNNa([gi,gj,gi◦gj,φ(i,j)])
          # gi is embedding of Pronoun
          # gj is embedding of A or B
          # gi◦gj is element-wise multiplication
          # φ(i,j) is the distance embedding
          if use_lingui_features:
            PA = Concatenate(axis=-1)([P, A, PA, dist_embeds[0]] + [dist_inputs[i] for i in [2,3,4,5,6]])
            PB = Concatenate(axis=-1)([P, B, PB, dist_embeds[1]] + [dist_inputs[i] for i in [7,8,9,10,11]])
          else:
            PA = Concatenate(axis=-1)([P, A, PA, dist_embeds[0]])
            PB = Concatenate(axis=-1)([P, B, PB, dist_embeds[1]])
          PA_score = self.ffnn(PA)
          PB_score = self.ffnn(PB)
          # Fix the Neither to score 0.
          score_e  = Lambda(lambda x: K.zeros_like(x))(PB_score)
          
          #Final Output
          output = Concatenate(axis=-1)([PA_score, PB_score, score_e]) # [Pronoun and A score, Pronoun and B score, Neither Score]
          output = Activation('softmax')(output)        
          model = Model(inputs+dist_inputs, output)
          
          return model

  

  if all_train:
    
    n_fold = 5
    n_run = 5
    num_test = pd.read_csv(input_tsv,sep='\t').shape[0]

    TTA_suffixes = \
    ['Alice_Kate_John_Michael',
    'Elizabeth_Mary_James_Henry',
    'Kate_Elizabeth_Michael_James',
    'Mary_Alice_Henry_John',
    'orig']

    pred_ensemble_end2end = np.zeros((num_test,3))

    for CASED in [False,True]:
      gc.collect()

      d_X_test = load_stage2_features(CASED)
      
      model = End2End_NCR(word_input_shape=d_X_test['orig'][0].shape[1], dist_shape=d_X_test['orig'][3].shape[1]).build()

      if CASED: 
        wts_prefix = path + 'wts/e2e-4_CASED_LARGE_Aug4_all_train_4400_Lingui_10_'
        ensemble_wts = [0.15, 0.2, 0.35, 0.2, 0.1]
      else:     
        wts_prefix = path + 'wts/e2e-4_LARGE_Aug4_all_train_4400_Lingui_10_'
        ensemble_wts = [0.25, 0.2, 0.3, 0.15, 0.1]

      pred_all_d = {} # to save 125 fold avg (for Test), 5 runs, 5 outer OOF, 5 inner early stop val
      for TTA_suffix in TTA_suffixes: pred_all_d[TTA_suffix] = np.zeros((num_test,3))     

      print('------ start inference ------')

      for run in tqdm(range(n_run)):  
        for fold in range(n_fold):
          for fold_inner in range(n_fold):
            wts = wts_prefix + f'{run}{fold}{fold_inner}.hdf5'
            model.load_weights(wts)
            for TTA_suffix in TTA_suffixes:   
              pred = model.predict(x = d_X_test[TTA_suffix], verbose = 0)    
              pred_all_d[TTA_suffix] += pred / n_fold / n_fold / n_run 


      pred_ensemble = np.zeros((num_test,3))    
      print(ensemble_wts)
      for i,TTA_suffix in enumerate(TTA_suffixes):    
        pred_ensemble += ensemble_wts[i]*pred_all_d[TTA_suffix]

      if not CASED: pred_ensemble_end2end += pred_ensemble * 0.4
      else:         pred_ensemble_end2end += pred_ensemble * 0.6


    assert pred_ensemble_end2end.sum(axis=1).min() > 0.999 and pred_ensemble_end2end.sum(axis=1).max() < 1.001    

  ## read stage2 sample submission and write output csv 

  sub = pd.read_csv(path+'input/sample_submission_stage_2.csv')


  sub_end2end = sub.copy()
  sub_end2end[['A','B','NEITHER']] = pred_ensemble_end2end

  out_csv_path = path + 'sub/sub_end2end_' + os.path.basename(input_tsv).split('.')[0] + '.csv'
  if os.path.exists(out_csv_path): os.remove(out_csv_path)
  sub_end2end.to_csv(out_csv_path, index=False)



  if not all_train:

    n_fold = 5
    n_run = 5
    num_test = pd.read_csv(input_tsv,sep='\t').shape[0]

    TTA_suffixes = \
    ['Alice_Kate_John_Michael',
    'Elizabeth_Mary_James_Henry',
    'Kate_Elizabeth_Michael_James',
    'Mary_Alice_Henry_John',
    'orig']

    pred_ensemble_end2endB = np.zeros((num_test,3))

    for CASED in [False,True]:
      gc.collect()

      d_X_test = load_stage2_features(CASED)
      
      model = End2End_NCR(word_input_shape=d_X_test['orig'][0].shape[1], dist_shape=d_X_test['orig'][3].shape[1]).build()

      if CASED: 
        wts_prefix = path + 'wts/e2e-4_CASED_LARGE_Aug4_sub_B_4400_'
        ensemble_wts = [0.2, 0.2, 0.4, 0.1, 0.1]
      else:     
        wts_prefix = path + 'wts/e2e-4_LARGE_Aug4_sub_B_4400_'
        ensemble_wts = [0.2, 0.2, 0.4, 0.0, 0.2]

      pred_all_d = {} 
      for TTA_suffix in TTA_suffixes: pred_all_d[TTA_suffix] = np.zeros((num_test,3))     

      print('------ start inference ------')

      for run in range(n_run):  
        for fold in range(n_fold):
          wts = wts_prefix + f'{run}{fold}.hdf5'
          model.load_weights(wts)
          for TTA_suffix in TTA_suffixes:   
            pred = model.predict(x = d_X_test[TTA_suffix], verbose = 0)    
            pred_all_d[TTA_suffix] += pred / n_fold / n_run         

      pred_ensemble = np.zeros((num_test,3))    
      print(ensemble_wts)
      for i,TTA_suffix in enumerate(TTA_suffixes):    
        pred_ensemble += ensemble_wts[i]*pred_all_d[TTA_suffix]

      if not CASED: pred_ensemble_end2endB += pred_ensemble * 0.4
      else:         pred_ensemble_end2endB += pred_ensemble * 0.6


    assert pred_ensemble_end2endB.sum(axis=1).min() > 0.999 and pred_ensemble_end2endB.sum(axis=1).max() < 1.001    

  ## read stage2 sample submission and write output csv 

  sub = pd.read_csv(path+'input/sample_submission_stage_2.csv')


  subB_end2end = sub.copy()
  subB_end2end[['A','B','NEITHER']] = pred_ensemble_end2endB

  out_csv_path = path + 'sub/subB_end2end_' + os.path.basename(input_tsv).split('.')[0] + '.csv'
  if os.path.exists(out_csv_path): os.remove(out_csv_path)
  subB_end2end.to_csv(out_csv_path, index=False)

  def parse_json(embeddings):
    '''
    Parses the embeddigns given by BERT, and suitably formats them to be passed to the MLP model

    Input: embeddings, a DataFrame containing contextual embeddings from BERT, as well as the labels for the classification problem
    columns: "emb_A": contextual embedding for the word A
            "emb_B": contextual embedding for the word B
            "emb_P": contextual embedding for the pronoun
            "label": the answer to the coreference problem: "A", "B" or "NEITHER"

    Output: X, a numpy array containing, for each line in the GAP file, the concatenation of the embeddings of the target words
            Y, a numpy array containing, for each line in the GAP file, the one-hot encoded answer to the coreference problem
    '''
    embeddings.sort_index(inplace = True) # Sorting the DataFrame, because reading from the json file messed with the order
    num_token = 3
    BS = 768 if not LARGE else 1024
    X = np.zeros((len(embeddings),num_token*BS)) 

    # Concatenate features
    for i in range(len(embeddings)):
      A = np.array(embeddings.loc[i,"emb_A"])
      B = np.array(embeddings.loc[i,"emb_B"])
      P = np.array(embeddings.loc[i,"emb_P"])
      X[i] = np.concatenate((A,B,P))
          
    return X


  def make_np_features_from_json(CASED = True,
                                LARGE = True,
                                MAX_SEQ_LEN = 256,
                                layer = None,
                                concat_lst = ["-3","-4"],
                                TTA_suffix = ''
                                ):  
    # single layer
    if concat_lst == None:
      suffix = ''
      if CASED: suffix += '_CASED'
      if LARGE: suffix += '_LARGE'   

      num_test = pd.read_csv(input_tsv,sep='\t').shape[0]
      n_chunk = int(np.ceil(num_test/1000))
      for i in range(n_chunk):     
        json_name = path + 'output/contextual_embeddings_' + os.path.basename(input_tsv).split('.')[0] + '_' + layer+ suffix +TTA_suffix+  f"_{i}" +'_fix_long_text.json'
        stage2_emb_chunk = pd.read_json(json_name).sort_index()
        if i==0: stage2_emb = stage2_emb_chunk.copy()
        else:    stage2_emb = pd.concat([stage2_emb,stage2_emb_chunk])
      stage2_emb = stage2_emb.reset_index(drop=True).copy()     
      X_stage2 = parse_json(stage2_emb)   
                                
    # concat, recursive
    else:   
      for this_layer in concat_lst:      
        # recursive
        X_stage2_layer = \
            make_np_features_from_json(CASED, LARGE, MAX_SEQ_LEN, this_layer, None, TTA_suffix)

        if this_layer==concat_lst[0]:
          X_stage2 = X_stage2_layer
        else:
          X_stage2 = np.concatenate((X_stage2,X_stage2_layer),axis=1)  
      
    return X_stage2               
  
  if all_train:
    pred_two_model_A = pred_ensemble_end2end
    pred_two_model_A = np.clip(pred_two_model_A, 0.005, None)

    sub_two_model = sub.copy()
    sub_two_model[['A','B','NEITHER']] = pred_two_model_A

    out_csv_path = path + 'sub/sub_two_model_' + os.path.basename(input_tsv).split('.')[0] + '.csv'
    if os.path.exists(out_csv_path): os.remove(out_csv_path)
    sub_two_model.to_csv(out_csv_path, index=False)
