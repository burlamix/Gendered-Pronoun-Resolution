import logging
import logging.config
import hltproject.utils.config as cutils

logging.config.dictConfig(
    cutils.load_logger_config_file())
logger = logging.getLogger ( __name__ )
logger.setLevel (logging.INFO)

def original_notebook_e2e ( all_train, CASED, path, dev_input_fname, test_input_fname, val_input_fname ):
  '''
    \param all_train, CASED original notebook parameters
    \param path path where to save weights
    \param dev_input_fname original input development set filename
    \param test_input_fname original input test set filename
    \param val_input_fname original input validation set filename
  '''
    
  logger.info ("model_7.original_notebook_e2e all_train {} CASED {}".format(all_train, CASED))

  import numpy as np # linear algebra
  import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
  import os
  import zipfile
  import gc
  from tqdm import tqdm as tqdm
  import re
  from glob import glob
  
  from sklearn.model_selection import KFold
  from sklearn.metrics import log_loss
  
  import subprocess

  #PATH CONSTANTS FOR CONSISTENCY WITH OTHER NOTEBOOKS 
  EMBEDDINGS_FILES_PREFIX = path + "/embeddings/" + '{}' + '_'
  LINGUI_CSV_FNAME = path + "/output/" + '{}' + "_lingui_df.csv"
  DIST_CSV_FNAME = path + '/output/' + '{}' + '_dist_df.csv'
  CORRECTED74_FNAME = path+'/input/gap-development-corrected-74.tsv'
  TEST_VAL85_FNAME = path+'/input/gap-test-val-85.tsv'
  
  dev_basename = os.path.basename (dev_input_fname).split('.')[0]
  test_basename = os.path.basename (test_input_fname).split('.')[0]
  val_basename = os.path.basename (val_input_fname).split('.')[0]

  use_lingui_features = True if all_train else False
  
  # dist features and linguistic features
  
  if all_train:  
    # leave 54 as sanity check
    np.random.seed(15)
    sanity_idx = np.random.choice(454,54,replace=False)
    val_idx = np.setdiff1d(np.arange(454),sanity_idx)  
    
  
  if not use_lingui_features:
  
    dev_dist_df = pd.read_csv(DIST_CSV_FNAME.format (dev_basename))
    test_dist_df =pd.read_csv(DIST_CSV_FNAME.format (test_basename))
    val_dist_df = pd.read_csv(DIST_CSV_FNAME.format (val_basename))
  
    if not all_train:
      new_dist_df = pd.concat([test_dist_df, val_dist_df]).reset_index(drop=True).copy()
      test_dist_df = dev_dist_df.copy()
    if all_train:
      new_dist_df = pd.concat([dev_dist_df,
                               test_dist_df, 
                               val_dist_df.iloc[val_idx]]).reset_index(drop=True).copy()
  
    print(new_dist_df.shape, test_dist_df.shape)
    
  if use_lingui_features:
    dev_lingui_df = pd.read_csv(LINGUI_CSV_FNAME.format(dev_basename))
    test_lingui_df =pd.read_csv(LINGUI_CSV_FNAME.format(test_basename))
    val_lingui_df = pd.read_csv(LINGUI_CSV_FNAME.format(val_basename))   
    
    dev_dist_df = pd.concat([pd.read_csv(DIST_CSV_FNAME.format (dev_basename))[['D_PA','D_PB']], dev_lingui_df], axis=1)
    test_dist_df =pd.concat([pd.read_csv(DIST_CSV_FNAME.format (test_basename))[['D_PA','D_PB']],test_lingui_df],axis=1)
    val_dist_df = pd.concat([pd.read_csv(DIST_CSV_FNAME.format (val_basename))[['D_PA','D_PB']], val_lingui_df], axis=1)
  
    if not all_train:
      new_dist_df  = pd.concat([test_dist_df, val_dist_df]).reset_index(drop=True).copy()
      test_dist_df = dev_dist_df.copy()
  
    if all_train:
      new_dist_df = pd.concat([dev_dist_df,
                               test_dist_df, 
                               val_dist_df.iloc[val_idx]]).reset_index(drop=True).copy()
      test_dist_df = val_dist_df.iloc[sanity_idx].reset_index(drop=True).copy()
  
    gc.collect()
  
  def create_input(embed_df, dist_df):
      
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
                  
          label = embed_df.loc[i, "label"]
          if label == "A": 
              all_label.append(0)
          elif label == "B": 
              all_label.append(1)
          else: 
              all_label.append(2)
  
      result_lst = [np.asarray(all_A), np.asarray(all_B), np.asarray(all_P),
                    np.expand_dims(np.asarray(all_dist_PA),axis=1),
                    np.expand_dims(np.asarray(all_dist_PB),axis=1)]
  
      if use_lingui_features:
        for col in dev_lingui_df.columns.values:
          result_lst.append(np.expand_dims(dist_df[col].values,axis=1))
              
      return result_lst, all_label
  
  # load previously extracted Bert features, orig and Aug
  
  LARGE = True
  
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
  new_emb_df_d = {} # dict for train embeddings (to be converted to features later after removing NA)
  
  for TTA_suffix in ['_'] + TTA_suffixes:
    gc.collect()
    
    logger.info ("reading embeddings datasets: fname: {}".format( EMBEDDINGS_FILES_PREFIX.format( dev_basename )+suffix+TTA_suffix+json_suffix ))

    df_dev1 = pd.read_json(EMBEDDINGS_FILES_PREFIX.format( dev_basename )+suffix+TTA_suffix+json_suffix).sort_index()
    df_test1= pd.read_json(EMBEDDINGS_FILES_PREFIX.format( test_basename )+suffix+TTA_suffix+json_suffix).sort_index()
    df_val =  pd.read_json(EMBEDDINGS_FILES_PREFIX.format( val_basename )+suffix+TTA_suffix+json_suffix).sort_index()  
    
    if not all_train:       
      new_emb_df0 = pd.concat([df_test1.sort_index(),\
                               df_val.sort_index()]).reset_index(drop=True).copy()
      test_emb0 = pd.concat([df_dev1.sort_index()]).reset_index(drop=True).copy()
  
    if all_train:  
      # leave 54 as sanity check
      np.random.seed(15)
      sanity_idx = np.random.choice(454,54,replace=False)
      val_idx = np.setdiff1d(np.arange(454),sanity_idx)     
      
      test_emb0 = df_val.iloc[sanity_idx].reset_index(drop=True).copy()  
      new_emb_df0 = pd.concat([df_dev1.sort_index(),
                              df_test1.sort_index(),
                              df_val.sort_index().iloc[val_idx]]).reset_index(drop=True).copy() 
  
    # put into dictionary
    key = 'orig' if TTA_suffix=='_' else TTA_suffix.strip('_')    
  
    X_test0, y_test = create_input(test_emb0, test_dist_df)    
    d_X_test[key] = X_test0.copy()
  
    new_emb_df_d[key] = new_emb_df0.copy()
  
  
  if all_train: 
    test_val_df = pd.concat([pd.read_table(CORRECTED74_FNAME)[['ID','A-coref','B-coref']],
                             pd.read_table(TEST_VAL85_FNAME)[['ID','A-coref','B-coref']].iloc[:2000],
                             pd.read_table(TEST_VAL85_FNAME)[['ID','A-coref','B-coref']].iloc[2000:].iloc[val_idx]]).reset_index(drop=True)
  else:
    test_val_df = pd.read_table(TEST_VAL85_FNAME)[['ID','A-coref','B-coref']].reset_index(drop=True)
    
    
  test_val_df['label'] = test_val_df.apply(lambda x:'A' if x['A-coref'] else ('B' if x['B-coref'] else 'Neither'),axis=1)
  
  logger.info ("number of wrong labels = {:d}".format((test_val_df.label!=new_emb_df_d['orig'].label).sum()))
  
  # fix labels
  new_emb_df_d['orig'].label = test_val_df.label
  
  assert (test_val_df.label!=new_emb_df_d['orig'].label).sum() == 0
  
  TTA_suffixes = [ 'Alice_Kate_John_Michael',
                   'Elizabeth_Mary_James_Henry',
                   'Kate_Elizabeth_Michael_James',
                   'Mary_Alice_Henry_John']
  
  for TTA_suffix in ['orig'] + TTA_suffixes:
    bad_rows = []
    for i in range(new_emb_df_d[TTA_suffix].shape[0]):
      for col in ['emb_A','emb_B','emb_P']:
        if None in new_emb_df_d[TTA_suffix].loc[i,col]:
          bad_rows.append(i)
          break
    logger.info (TTA_suffix + ' bad_rows =' + str(bad_rows))  
    
  ## remove None
  
  # remove bad rows in (common) dist df
  new_dist_df = new_dist_df.drop(bad_rows).reset_index(drop=True)
  
  # and remove them in all emb df
  for TTA_suffix in ['orig'] + TTA_suffixes:  
      
    new_emb_df_d[TTA_suffix] = new_emb_df_d[TTA_suffix].drop(bad_rows).reset_index(drop=True)
    
    assert new_emb_df_d[TTA_suffix].shape[0]==new_dist_df.shape[0]
  
    print(TTA_suffix, new_emb_df_d[TTA_suffix].shape)     
  
  X_train_d = {}
  for TTA_suffix in TTA_suffixes + ['orig']: 
    X_train0, y_train = create_input(new_emb_df_d[TTA_suffix], new_dist_df)
    X_train_d[TTA_suffix] = X_train0.copy()
  
    
  y_one_hot = np.zeros((len(y_test), 3))
  for i in range(len(y_test)): y_one_hot[i, y_test[i]] = 1
    
  y_train_one_hot = np.zeros((len(y_train), 3))
  for i in range(len(y_train)): y_train_one_hot[i, y_train[i]] = 1
    
  gc.collect()  
  y_train_one_hot.shape, y_one_hot.shape
  
  import keras.backend as K
  import keras
  from keras.layers import Input, Dense, Embedding, Activation, Dropout, Flatten, Multiply, Concatenate, Lambda
  from keras.models import Model, Sequential
  from keras import optimizers
  from keras import callbacks
  
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
            dist_inputs = [Input((1,)) for i in range(len(X_test0)-3)]
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
  
  batch_size = 128
  lr = 3e-5 if all_train else 1e-4
  patience_orig = 100 
  patience = patience_orig
  epochs = 1000
  n_fold = 5
  
  suffix = layer
  if CASED: suffix += '_CASED'
  if LARGE: suffix += '_LARGE'
    
    
  TTA_suffixes = \
  ['orig',
   'Alice_Kate_John_Michael',
   'Elizabeth_Mary_James_Henry',
   'Kate_Elizabeth_Michael_James',
   'Mary_Alice_Henry_John']
  
  Aug_suffixes = \
  ['Alice_Kate_John_Michael',
   'Elizabeth_Mary_James_Henry',
   'Kate_Elizabeth_Michael_James',
   'Mary_Alice_Henry_John']  
  
  if len(Aug_suffixes)==3: suffix += '_Aug3'
  if len(Aug_suffixes)==4: suffix += '_Aug4'  
  suffix += '_all_train' if all_train else '_sub_B'
    
  suffix += '_4400'  
  suffix += '_Lingui_10'
    
  logger.info(suffix)
  
  num_test = len(y_test)
  num_train = len(y_train)
  print(num_train,num_test)
  
  
  #####  all_train
  # pred_oof, sub_all_d: OOF
  # pred_all_d, sub_df : sanity
  
  #####  sub_B
  # pred_all_d, sub_df : dev
  
  if all_train:
  
    for run in range(0,5):  
      gc.collect()
  
      if all_train:
        sub_all = pd.concat([pd.read_table(path+'/input/gap-development.tsv',usecols=['ID']),
                             pd.read_table(path+'/input/gap-test.tsv',usecols=['ID']),
                             pd.read_table(path+'/input/gap-validation.tsv',usecols=['ID']).iloc[val_idx]]).\
                  reset_index(drop=True).drop(bad_rows)
        sub_all['A']=0; sub_all['B']=0; sub_all['NEITHER']=0
        sub_all_d = {}
        for TTA_suffix in TTA_suffixes: sub_all_d[TTA_suffix] = sub_all.copy()  
  
      pred_all_d = {} # to save 25 fold avg (for Test), 5 outer OOF, 5 inner early stop
      for TTA_suffix in TTA_suffixes: pred_all_d[TTA_suffix] = np.zeros((num_test,3))        
  
      # outer 5 fold: OOF fold. 4/5 train, 1/5 OOF pred  
      kfold = KFold(n_splits=n_fold, shuffle=True, random_state=3)
      for fold_n, (train_fold_index, oof_val_index) in enumerate(kfold.split(X_train_d['orig'][0])):
          y_train_fold  = np.asarray(y_train)[train_fold_index]
         # y_oof_val = np.asarray(y_train)[oof_val_index]      
  
          X_train_fold_d = {}; X_oof_val_d = {}
          for TTA_suffix in TTA_suffixes: 
            X_train_fold_d[TTA_suffix] = [inputs[train_fold_index] for inputs in X_train_d[TTA_suffix]]    
            X_oof_val_d[TTA_suffix] = [inputs[oof_val_index] for inputs in X_train_d[TTA_suffix]]    
  
          # inner 5 fold: train and early-stop val fold.
          kfold_inner = KFold(n_splits=n_fold, shuffle=True, random_state=5)
          for fold_n_inner, (train_index, valid_index) in enumerate(kfold_inner.split(X_train_fold_d['orig'][0])):
  
              X_tr  = [inputs[train_index] for inputs in X_train_fold_d['orig']] 
              # X_tr_orig = X_tr.copy()
              X_val = [inputs[valid_index] for inputs in X_train_fold_d['orig']]   
              y_tr  = np.asarray(y_train_fold)[train_index]
              y_val = np.asarray(y_train_fold)[valid_index]
  
              # train augmentation
              if len(Aug_suffixes)>1: 
                patience = np.ceil(patience_orig / (1+len(Aug_suffixes)))
                for k in Aug_suffixes:
                  X_tr0 = [inputs[train_index] for inputs in X_train_fold_d[k]]
  
                  for i in range(len(X_tr)): X_tr[i] = np.concatenate((X_tr[i], X_tr0[i]),axis=0)
                  y_tr = np.concatenate((y_tr, y_tr),axis=0)
  
                arr = np.arange(X_tr[0].shape[0])
                np.random.shuffle(arr)
                X_tr = [X_tr0[arr,:] for X_tr0 in X_tr]
                y_tr = y_tr[arr]      
                print(X_tr[0].shape, y_tr.shape, X_val[0].shape, y_val.shape)  
  
              model = End2End_NCR(word_input_shape=X_train_fold_d['orig'][0].shape[1], dist_shape=X_train_fold_d['orig'][3].shape[1]).build()
              model.compile(optimizer=optimizers.Adam(lr=lr), loss="sparse_categorical_crossentropy")
              file_path = path + '/wts/e2e' + suffix + "_{}{}{}.hdf5".format(run,fold_n,fold_n_inner)
              check_point = callbacks.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 0, save_weights_only=True, save_best_only = True, mode = "min")
              early_stop = callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience=patience, restore_best_weights = True)    
              model.fit(X_tr, y_tr, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=0,
                        shuffle=True, callbacks = [check_point, early_stop])
  
              for TTA_suffix in TTA_suffixes:
                pred = model.predict(x = d_X_test[TTA_suffix], verbose = 0)
                pred_all_d[TTA_suffix] += pred / n_fold / n_fold 
                if all_train:
                  pred_oof = model.predict(x = X_oof_val_d[TTA_suffix], verbose=0)      
                  sub_all_d[TTA_suffix].loc[sub_all_d[TTA_suffix].index[oof_val_index],['A','B','NEITHER']] += pred_oof / n_fold
  
  
      for TTA_suffix in TTA_suffixes:    
        # for Test
        if all_train:
          sub_df = pd.read_csv(path+'/input/gap-validation.tsv',sep='\t').iloc[sanity_idx][['ID']]
          sub_df['A'] = 1/3; sub_df['B'] = 1/3; sub_df['NEITHER'] = 1/3 
        else:
          sub_df = pd.read_csv(path+'/input/sample_submission_stage_1.csv')
  
        sub_df.loc[:,['A','B','NEITHER']] = pred_all_d[TTA_suffix]      
        sub_df.to_csv(path+'/sub/end2end'+suffix+'_'+TTA_suffix+'_run{:d}_{:.5f}.csv'.format(run,log_loss(y_one_hot, pred_all_d[TTA_suffix])), index=False)        
        logger.info(f'run{run} {TTA_suffix} ' + "{:d}folds {:.5f}".format(n_fold, log_loss(y_one_hot, pred_all_d[TTA_suffix]))) # Calculate the log loss 
  
        if all_train:
          sub_all_d[TTA_suffix].to_csv(path+'/sub/oof'+suffix+'_'+TTA_suffix+'_run{:d}_{:.5f}.csv'.format(run,log_loss(y_train_one_hot, sub_all_d[TTA_suffix].loc[:,['A','B','NEITHER']].values)), index=False)
          logger.info(f'run{run} {TTA_suffix} ' + "{:d}folds OOF ================= {:.5f}".format(n_fold, log_loss(y_train_one_hot, sub_all_d[TTA_suffix].loc[:,['A','B','NEITHER']]))) # Calculate the log loss    
  
  #####  all_train
  # pred_oof, sub_all_d: OOF
  # pred_all_d, sub_df : sanity
  
  #####  sub_B
  # pred_all_d, sub_df : dev
  
  if not all_train:
  
    for run in range(0,5):  
      gc.collect()
  
      pred_all_d = {} 
      for TTA_suffix in TTA_suffixes: pred_all_d[TTA_suffix] = np.zeros((num_test,3))        
  
      kfold = KFold(n_splits=n_fold, shuffle=True, random_state=3)
      for fold_n, (train_index, valid_index) in enumerate(kfold.split(X_train_d['orig'][0])):   
  
          X_tr  = [inputs[train_index] for inputs in X_train_d['orig']] 
          # X_tr_orig = X_tr.copy()
          X_val = [inputs[valid_index] for inputs in X_train_d['orig']]   
          y_tr  = np.asarray(y_train)[train_index]
          y_val = np.asarray(y_train)[valid_index]
  
          # train augmentation
          if len(Aug_suffixes)>1: 
            patience = np.ceil(patience_orig / (1+len(Aug_suffixes)))
            for k in Aug_suffixes:
              X_tr0 = [inputs[train_index] for inputs in X_train_d[k]]
  
              for i in range(len(X_tr)): X_tr[i] = np.concatenate((X_tr[i], X_tr0[i]),axis=0)
              y_tr = np.concatenate((y_tr, y_tr),axis=0)
  
            arr = np.arange(X_tr[0].shape[0])
            np.random.shuffle(arr)
            X_tr = [X_tr0[arr,:] for X_tr0 in X_tr]
            y_tr = y_tr[arr]      
            # print(X_tr[0].shape, y_tr.shape, X_val[0].shape, y_val.shape)  
  
          model = End2End_NCR(word_input_shape=X_train_d['orig'][0].shape[1], dist_shape=X_train_d['orig'][3].shape[1]).build()
          model.compile(optimizer=optimizers.Adam(lr=lr), loss="sparse_categorical_crossentropy")
          file_path = path + '/wts/e2e' + suffix + "_{}{}.hdf5".format(run,fold_n)
          check_point = callbacks.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 0, save_weights_only=True, save_best_only = True, mode = "min")
          early_stop = callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience=patience, restore_best_weights = True)    
          model.fit(X_tr, y_tr, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=0,
                    shuffle=True, callbacks = [check_point, early_stop])
  
          for TTA_suffix in TTA_suffixes:
            pred = model.predict(x = d_X_test[TTA_suffix], verbose = 0)
            pred_all_d[TTA_suffix] += pred / n_fold  
  
      for TTA_suffix in TTA_suffixes:    
        sub_df = pd.read_csv(path+'/input/sample_submission_stage_1.csv')
  
        sub_df.loc[:,['A','B','NEITHER']] = pred_all_d[TTA_suffix]      
        sub_df.to_csv(path+'/sub/end2end'+suffix+'_'+TTA_suffix+'_run{:d}_{:.5f}.csv'.format(run,log_loss(y_one_hot, pred_all_d[TTA_suffix])), index=False)        
        print(f'run{run} {TTA_suffix} ' + "{:d}folds {:.5f}".format(n_fold, log_loss(y_one_hot, pred_all_d[TTA_suffix]))) # Calculate the log loss 
  
  
  
  