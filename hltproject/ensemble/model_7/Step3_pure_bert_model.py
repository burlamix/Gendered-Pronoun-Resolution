path = 'drive/My Drive/pronoun/'

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import zipfile
import gc
from tqdm import tqdm as tqdm
import re
from glob import glob
import shutil

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

subprocess.call("wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv", shell=True)
subprocess.call("wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv", shell=True)
subprocess.call("wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", shell=True)

# move json files from Drive to local for fast read (only needed for Colab)
for x in tqdm([x for x in sorted(glob('drive/My Drive/pronoun/output/contextual_embeddings_gap*_fix_long_text.json')) if 'LARGE' in x]):
  shutil.copy2(x, os.path.basename(x)) 

all_train = False
CASED = False

def parse_json(embeddings, 
               debug=False,
               overwrite_dev_labels = None, # 0 first half, 1 second half
               overwrite_test_labels = None, # 0 first half, 1 second half
               overwrite_val_labels = False
              ):
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
	Y = np.zeros((len(embeddings), 3))

	# Concatenate features
	for i in range(len(embeddings)):
		A = np.array(embeddings.loc[i,"emb_A"])
		B = np.array(embeddings.loc[i,"emb_B"])
		P = np.array(embeddings.loc[i,"emb_P"])
		X[i] = np.concatenate((A,B,P))

	# One-hot encoding for labels
	for i in range(len(embeddings)):
		label = embeddings.loc[i,"label"]
		if label == "A":  Y[i,:] = [1,0,0]
		elif label == "B":Y[i,:] = [0,1,0]
		else:             Y[i,:] = [0,0,1]
      
	if overwrite_dev_labels==0:
		print('============ USING gap-development-corrected-74 LABELS !!! ============')
		assert len(embeddings) == 1000
		cnt = 0
		for i in range(1000):
			df = pd.read_csv('drive/My Drive/pronoun/input/gap-development-corrected-74.tsv',sep='\t')
			label = 'A' if df.loc[i,"A-coref"] else ('B' if df.loc[i,'B-coref'] else 'Neither')
			if label!=embeddings.loc[i,"label"]: cnt +=1
			if label == "A":  Y[i,:] = [1,0,0]
			elif label == "B":Y[i,:] = [0,1,0]
			else:             Y[i,:] = [0,0,1]  
		print('corrected {:d} labels'.format(cnt))        
	elif overwrite_dev_labels==1:
		print('============ USING gap-development-corrected-74 LABELS !!! ============')    
		assert len(embeddings) == 1000
		cnt = 0    
		for i in range(1000):
			df = pd.read_csv('drive/My Drive/pronoun/input/gap-development-corrected-74.tsv',sep='\t')
			label = 'A' if df.loc[1000+i,"A-coref"] else ('B' if df.loc[1000+i,'B-coref'] else 'Neither')
			if label!=embeddings.loc[i,"label"]: cnt +=1      
			if label == "A":  Y[i,:] = [1,0,0]
			elif label == "B":Y[i,:] = [0,1,0]
			else:             Y[i,:] = [0,0,1]          
		print('corrected {:d} labels'.format(cnt))        
    
	if overwrite_test_labels==0:
		print('============ USING gap-test-val-85 LABELS (test0) !!! ============')
		assert len(embeddings) == 1000
		cnt = 0
		for i in range(1000):
			df = pd.read_csv('drive/My Drive/pronoun/input/gap-test-val-85.tsv',sep='\t')
			label = 'A' if df.loc[i,"A-coref"] else ('B' if df.loc[i,'B-coref'] else 'Neither')
			if label!=embeddings.loc[i,"label"]: cnt +=1
			if label == "A":  Y[i,:] = [1,0,0]
			elif label == "B":Y[i,:] = [0,1,0]
			else:             Y[i,:] = [0,0,1]  
		print('corrected {:d} labels'.format(cnt))        
	elif overwrite_test_labels==1:
		print('============ USING gap-test-val-85 LABELS (test1) !!! ============')    
		assert len(embeddings) == 1000
		cnt = 0    
		for i in range(1000):
			df = pd.read_csv('drive/My Drive/pronoun/input/gap-test-val-85.tsv',sep='\t')
			label = 'A' if df.loc[1000+i,"A-coref"] else ('B' if df.loc[1000+i,'B-coref'] else 'Neither')
			if label!=embeddings.loc[i,"label"]: cnt +=1      
			if label == "A":  Y[i,:] = [1,0,0]
			elif label == "B":Y[i,:] = [0,1,0]
			else:             Y[i,:] = [0,0,1]             
		print('corrected {:d} labels'.format(cnt))    
    
	if overwrite_val_labels:
		print('============ USING gap-test-val-85 LABELS (val) !!! ============')
		assert len(embeddings) == 454
		cnt = 0
		for i in range(454):
			df = pd.read_csv('drive/My Drive/pronoun/input/gap-test-val-85.tsv',sep='\t')
			label = 'A' if df.loc[2000+i,"A-coref"] else ('B' if df.loc[2000+i,'B-coref'] else 'Neither')
			if label!=embeddings.loc[i,"label"]: cnt +=1
			if label == "A":  Y[i,:] = [1,0,0]
			elif label == "B":Y[i,:] = [0,1,0]
			else:             Y[i,:] = [0,0,1]       
		print('corrected {:d} labels'.format(cnt))      
    
	return X, Y

def get_json_names(CASED = True,
                   LARGE = True,
                   MAX_SEQ_LEN = 256,
                   layer = None,
                   concat_lst = ["-3","-4"],
                   TTA_suffix = ''):
  
  suffix = layer
  if CASED: suffix += '_CASED'
  if LARGE: suffix += '_LARGE'    
    
  suffix += ('_'+TTA_suffix) if TTA_suffix!='' else ''
  
  json_suffix = '_fix_long_text.json'
  
  if LARGE:
    json_names = ['contextual_embeddings_gap_development_'+suffix+'_1'+json_suffix,
                  'contextual_embeddings_gap_development_'+suffix+'_2'+json_suffix,
                  'contextual_embeddings_gap_test_'+suffix+'_1'+json_suffix,
                  'contextual_embeddings_gap_test_'+suffix+'_2'+json_suffix,
                  'contextual_embeddings_gap_validation_'+suffix+json_suffix]            
  else:
    raise Exception('Not implemented')
    
  print(json_names)
  return json_names


def make_np_features_from_json(CASED = True,
                               LARGE = True,
                               MAX_SEQ_LEN = 256,
                               layer = None,
                               concat_lst = ["-3","-4"],
                               TTA_suffix = '',
                               all_train = False):  
  # single layer
  if concat_lst == None:
    json_names = get_json_names(CASED, LARGE, MAX_SEQ_LEN, layer, concat_lst, TTA_suffix)
    validation = pd.read_json(json_names[-1])
    X_validation, Y_validation = parse_json(validation, overwrite_val_labels = True)

    development = pd.read_json(json_names[0])
    X_development1, Y_development1 = parse_json(development, overwrite_dev_labels=0 if all_train else None)
    development = pd.read_json(json_names[1])
    X_development2, Y_development2 = parse_json(development, overwrite_dev_labels=1 if all_train else None)

    X_development = np.concatenate((X_development1,X_development2))
    Y_development = np.concatenate((Y_development1,Y_development2))

    test = pd.read_json(json_names[2])
    X_test1, Y_test1 = parse_json(test, overwrite_test_labels = 0)
    test = pd.read_json(json_names[3])
    X_test2, Y_test2 = parse_json(test, overwrite_test_labels = 1)
    X_test = np.concatenate((X_test1,X_test2))
    Y_test = np.concatenate((Y_test1,Y_test2))    
      
    ## remove NaN rows, and combine train data
  
    if all_train:
      # train: 4400
      # sanity: 454
      np.random.seed(15)
      sanity_idx = np.random.choice(454,54,replace=False)
      val_idx = np.setdiff1d(np.arange(454),sanity_idx)     
      
      X_train = np.concatenate((X_development, X_test, X_validation[val_idx,:]), axis = 0).copy()
      Y_train = np.concatenate((Y_development, Y_test, Y_validation[val_idx,:]), axis = 0).copy()
      X_development = X_validation[sanity_idx,:]
      Y_development = Y_validation[sanity_idx,:]
      
      remove_development = [row for row in range(len(X_development)) if np.sum(np.isnan(X_development[row]))]
      print('remove_development: ' + str(remove_development))
      num_token = 3
      num_concat = len(concat_lst) if concat_lst != None else 1
      BS = 1024 if LARGE else 768
      X_development[remove_development] = np.zeros(num_token*BS*num_concat)            

    else:
      # train: test 2000 + val 454
      # predict: dev 2000
      # We want predictions for all development rows. So instead of removing rows, make them 0
      remove_development = [row for row in range(len(X_development)) if np.sum(np.isnan(X_development[row]))]
      print('remove_development: ' + str(remove_development))
      num_token = 3
      num_concat = len(concat_lst) if concat_lst != None else 1
      BS = 1024 if LARGE else 768
      X_development[remove_development] = np.zeros(num_token*BS*num_concat)

      # Will train on data from the gap-test and gap-validation files, in total 2454 rows
      X_train = np.concatenate((X_test, X_validation), axis = 0)
      Y_train = np.concatenate((Y_test, Y_validation), axis = 0)       
         
  # concat, recursive
  else:   
    for this_layer in concat_lst:      
      # recursive
      X_train_layer, Y_train_layer, X_development_layer, Y_development_layer = \
          make_np_features_from_json(CASED, LARGE, MAX_SEQ_LEN, this_layer, None, TTA_suffix, all_train)

      if this_layer==concat_lst[0]:
        X_development, Y_development = X_development_layer, Y_development_layer
        X_train, Y_train = X_train_layer, Y_train_layer
      else:
        X_development = np.concatenate((X_development,X_development_layer),axis=1)
        X_train = np.concatenate((X_train,X_train_layer),axis=1)  
    
  return X_train, Y_train, X_development, Y_development       
        

%%time

LARGE = True
concat_lst = ["-3","-4"]
layer = None # "-3"
MAX_SEQ_LEN = 256

TTA_suffixes = [\
                 'Alice_Kate_John_Michael',
                 'Elizabeth_Mary_James_Henry',
                 'Kate_Elizabeth_Michael_James',
                 'Mary_Alice_Henry_John']

d_XY = {}

for TTA_suffix in ['orig'] + TTA_suffixes:
  this_d = {}
  
  this_d['X_train'],this_d['Y_train'],this_d['X_dev'],this_d['Y_dev'] = \
          make_np_features_from_json(CASED = CASED,
                                     LARGE = LARGE,
                                     MAX_SEQ_LEN = MAX_SEQ_LEN,
                                     layer = layer,
                                     concat_lst = concat_lst,
                                     TTA_suffix = '' if TTA_suffix=='orig' else TTA_suffix,
                                     all_train=all_train) 
  print(this_d['X_train'].shape, this_d['Y_train'].shape, this_d['X_dev'].shape, this_d['Y_dev'].shape)
  
  d_XY[TTA_suffix] = this_d

print(d_XY['orig']['X_train'].shape, d_XY['orig']['Y_train'].shape, d_XY['orig']['X_dev'].shape, d_XY['orig']['Y_dev'].shape)  


# remove missing rows in both original and each Aug

index_train = list(range(d_XY['orig']['X_train'].shape[0]))
remove_train = [row for row in range(len(d_XY['orig']['X_train'])) if np.sum(np.isnan(d_XY['orig']['X_train'][row]))]
d_XY['orig']['X_train'] = np.delete(d_XY['orig']['X_train'], remove_train, 0)
d_XY['orig']['Y_train'] = np.delete(d_XY['orig']['Y_train'], remove_train, 0)

# which rows are left from 0 to 2453
index_train = np.delete(index_train, remove_train, 0)

print("removed: ", remove_train, len(remove_train))  

for k,v in d_XY.items():
  if k=='orig':
    continue
  remove_train_aug = [row for row in range(len(v['X_train'])) if np.sum(np.isnan(v['X_train'][row]))]
  print(k, remove_train_aug)

  # remove rows that are also missing in orignal
  v['X_train'] = np.delete(v['X_train'], remove_train, 0)
  v['Y_train'] = np.delete(v['Y_train'], remove_train, 0)

# sanity check
print('------- sanity check ---------')
for k,v in d_XY.items():
  print(k, v['X_train'].shape, v['Y_train'].shape, v['X_dev'].shape, v['Y_dev'].shape)  

print(d_XY['orig']['X_train'].shape, d_XY['orig']['Y_train'].shape, d_XY['orig']['X_dev'].shape, d_XY['orig']['Y_dev'].shape)   

from keras import backend, models, layers, initializers, regularizers, constraints, optimizers
from keras import callbacks as kc
from keras import optimizers as ko

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import log_loss
import time


def build_mlp_model(input_shape):
	X_input = layers.Input(input_shape)

	# First dense layer
	X = layers.Dense(dense_layer_sizes[0], name = 'dense0')(X_input)
	X = layers.BatchNormalization(name = 'bn0')(X)
	X = layers.Activation('relu')(X)
	X = layers.Dropout(dropout_rate, seed = 7)(X)

	# Second dense layer
	if len(dense_layer_sizes)==2:
		X = layers.Dense(dense_layer_sizes[1], name = 'dense1')(X)
		X = layers.BatchNormalization(name = 'bn1')(X)
		X = layers.Activation('relu')(X)
		X = layers.Dropout(dropout_rate, seed = 9)(X)

	# Output layer
	X = layers.Dense(3, name = 'output', kernel_regularizer = regularizers.l2(lambd))(X)
	X = layers.Activation('softmax')(X)

	# Create model
	model = models.Model(input = X_input, output = X, name = "classif_model")
	return model

loss = "categorical_crossentropy"

dense_layer_sizes = [64]
if concat_lst != None:
  dense_layer_sizes = [512,32]
dropout_rate = 0.6
learning_rate = 0.001
n_fold = 5
batch_size = 32
epochs = 1000
patience_orig = 60 if all_train else 100
patience = patience_orig
lambd = 0.1 # L2 regularization

np.random.seed(15)
sanity_idx = np.random.choice(454,54,replace=False)
val_idx = np.setdiff1d(np.arange(454),sanity_idx)
bad_rows = remove_train

suffix = 'pure_bert_' + ''.join(concat_lst)
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
#  'Elizabeth_Mary_James_Henry',
 'Kate_Elizabeth_Michael_James',
 'Mary_Alice_Henry_John']

if len(Aug_suffixes)==3: suffix += '_Aug3'
if len(Aug_suffixes)==4: suffix += '_Aug4'  
suffix += '_all_train' if all_train else '_sub_B'
  
suffix += '_4400'

print(suffix)  

num_test = d_XY['orig']['X_dev'].shape[0]
num_train = d_XY['orig']['X_train'].shape[0]
print(num_train,num_test)

%%time

#####  all_train
# pred_oof, sub_all_d: OOF
# pred_all_d, sub_df : sanity

#####  sub_B
# pred_all_d, sub_df : dev

if all_train:

  pd.DataFrame(columns=['a','b']).to_csv(path+'sub/tmp.csv') # testing drive connection

  for run in range(0,1):
    import gc; gc.collect()

    sub_all = pd.concat([pd.read_table(path+'input/gap-development.tsv',usecols=['ID']),
                         pd.read_table(path+'input/gap-test.tsv',usecols=['ID']),
                         pd.read_table(path+'input/gap-validation.tsv',usecols=['ID']).iloc[val_idx]]).\
              reset_index(drop=True).drop(bad_rows)
    assert sub_all.shape[0]==num_train
    sub_all['A']=0; sub_all['B']=0; sub_all['NEITHER']=0
    sub_all_d = {}
    for TTA_suffix in TTA_suffixes: sub_all_d[TTA_suffix] = sub_all.copy()  

    pred_all_d = {} # to save 25 fold avg (for Test), 5 outer OOF, 5 inner early stop
    for TTA_suffix in TTA_suffixes: pred_all_d[TTA_suffix] = np.zeros((num_test,3))        

    # outer 5 fold: OOF fold. 4/5 train, 1/5 OOF pred  
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=3)
    for fold_n, (train_fold_index, oof_val_index) in enumerate(kfold.split(d_XY['orig']['X_train'])):
      y_train_fold= d_XY['orig']['Y_train'][train_fold_index,:]

      X_train_fold_d = {}; X_oof_val_d = {}
      for TTA_suffix in TTA_suffixes: 
        X_train_fold_d[TTA_suffix]= d_XY[TTA_suffix]['X_train'][train_fold_index,:]   
        X_oof_val_d[TTA_suffix]   = d_XY[TTA_suffix]['X_train'][oof_val_index,:]   

      # inner 5 fold: train and early-stop val fold.
      kfold_inner = KFold(n_splits=n_fold, shuffle=True, random_state=5)
      for fold_n_inner, (train_index, valid_index) in enumerate(kfold_inner.split(X_train_fold_d['orig'])):        

        X_tr  = X_train_fold_d['orig'][train_index,:]
        X_tr_orig = X_tr.copy()
        X_val = X_train_fold_d['orig'][valid_index,:]
        y_tr  = y_train_fold[train_index,:]
        y_val = y_train_fold[valid_index,:]              

        # train augmentation
        if len(Aug_suffixes)>1: 
          patience = np.ceil(patience_orig / (1+len(Aug_suffixes)))

          for k in Aug_suffixes:
            X_tr = np.concatenate((X_tr, X_train_fold_d[k][train_index,:]),axis=0)
            y_tr = np.concatenate((y_tr, y_tr),axis=0)

          arr = np.arange(X_tr.shape[0])
          np.random.shuffle(arr)
          X_tr = X_tr[arr,:] 
          y_tr = y_tr[arr,:]  
          print(X_tr.shape, y_tr.shape, X_val.shape, y_val.shape)

        # Define the model, re-initializing for each fold
        classif_model = build_mlp_model([X_tr.shape[1]])
        classif_model.compile(optimizer = optimizers.Adam(lr = learning_rate), 
                              loss = loss)
        file_path = path + 'wts/pure_bert' + suffix + "_{}{}{}.hdf5".format(run,fold_n,fold_n_inner)
        callbacks = [kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 0, save_best_only = True, mode = "min"),
                     kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights = True)]

        # train the model
        classif_model.fit(x = X_tr, 
                          y = y_tr, 
                          epochs = epochs, 
                          batch_size = batch_size, 
                          callbacks = callbacks, 
                          validation_data = (X_val, y_val), 
                          verbose = 0)

        for TTA_suffix in TTA_suffixes:
          pred = classif_model.predict(x = d_XY[TTA_suffix]['X_dev'], verbose = 0)
          pred_oof = classif_model.predict(x = X_oof_val_d[TTA_suffix], verbose=0)      
          sub_all_d[TTA_suffix].loc[sub_all_d[TTA_suffix].index[oof_val_index],['A','B','NEITHER']] += pred_oof / n_fold
          pred_all_d[TTA_suffix] += pred / n_fold / n_fold    

    for TTA_suffix in TTA_suffixes:    
      # for Test
      sub_df = pd.read_csv(path+'input/gap-validation.tsv',sep='\t').iloc[sanity_idx][['ID']]
      sub_df['A'] = 1/3; sub_df['B'] = 1/3; sub_df['NEITHER'] = 1/3 

      sub_df.loc[:,['A','B','NEITHER']] = pred_all_d[TTA_suffix]      
      sub_df.to_csv(path+'sub/test_'+suffix+'_'+TTA_suffix+'_run{:d}_{:.5f}.csv'.format(run,log_loss(d_XY['orig']['Y_dev'], pred_all_d[TTA_suffix])), index=False)        
      print(f'run{run} {TTA_suffix} ' + "{:d}folds {:.5f}".format(n_fold, log_loss(d_XY['orig']['Y_dev'], pred_all_d[TTA_suffix]))) # Calculate the log loss 

      sub_all_d[TTA_suffix].to_csv(path+'sub/oof_'+suffix+'_'+TTA_suffix+'_run{:d}_{:.5f}.csv'.format(run,log_loss(d_XY['orig']['Y_train'], sub_all_d[TTA_suffix].loc[:,['A','B','NEITHER']].values)), index=False)
      print(f'run{run} {TTA_suffix} ' + "{:d}folds OOF ================= {:.5f}".format(n_fold, log_loss(d_XY['orig']['Y_train'], sub_all_d[TTA_suffix].loc[:,['A','B','NEITHER']]))) # Calculate the log loss    

%%time

#####  all_train
# pred_oof, sub_all_d: OOF
# pred_all_d, sub_df : sanity

#####  sub_B
# pred_all_d, sub_df : dev

if not all_train:

  pd.DataFrame(columns=['a','b']).to_csv(path+'sub/tmp.csv') # testing drive connection

  for run in range(0,5):
    import gc; gc.collect()

    pred_all_d = {} # to save 25 fold avg (for Test), 5 outer OOF, 5 inner early stop
    for TTA_suffix in TTA_suffixes: pred_all_d[TTA_suffix] = np.zeros((num_test,3))        

    # outer 5 fold: OOF fold. 4/5 train, 1/5 OOF pred  
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=3)
    for fold_n, (train_index, valid_index) in enumerate(kfold.split(d_XY['orig']['X_train'])):     

      X_tr  = d_XY['orig']['X_train'][train_index,:]
      X_tr_orig = X_tr.copy()
      X_val = d_XY['orig']['X_train'][valid_index,:]
      y_tr  = d_XY['orig']['Y_train'][train_index,:]
      y_val = d_XY['orig']['Y_train'][valid_index,:]              

      # train augmentation
      if len(Aug_suffixes)>1: 
        patience = np.ceil(patience_orig / (1+len(Aug_suffixes)))

        for k in Aug_suffixes:
          X_tr = np.concatenate((X_tr, d_XY[k]['X_train'][train_index,:]),axis=0)
          y_tr = np.concatenate((y_tr, y_tr),axis=0)

        arr = np.arange(X_tr.shape[0])
        np.random.shuffle(arr)
        X_tr = X_tr[arr,:] 
        y_tr = y_tr[arr,:]  
        print(X_tr.shape, y_tr.shape, X_val.shape, y_val.shape)

      # Define the model, re-initializing for each fold
      classif_model = build_mlp_model([X_tr.shape[1]])
      classif_model.compile(optimizer = optimizers.Adam(lr = learning_rate), 
                            loss = loss)
      file_path = path + 'wts/pure_bert' + suffix + "_{}{}.hdf5".format(run,fold_n)
      callbacks = [kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 0, save_best_only = True, mode = "min"),
                   kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights = True)]

      # train the model
      classif_model.fit(x = X_tr, 
                        y = y_tr, 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        callbacks = callbacks, 
                        validation_data = (X_val, y_val), 
                        verbose = 0)

      for TTA_suffix in TTA_suffixes:
        pred = classif_model.predict(x = d_XY[TTA_suffix]['X_dev'], verbose = 0)
        pred_all_d[TTA_suffix] += pred / n_fold    

    for TTA_suffix in TTA_suffixes:    
      sub_df = pd.read_csv(path+'input/sample_submission_stage_1.csv')

      sub_df.loc[:,['A','B','NEITHER']] = pred_all_d[TTA_suffix]      
      sub_df.to_csv(path+'sub/test_'+suffix+'_'+TTA_suffix+'_run{:d}_{:.5f}.csv'.format(run,log_loss(d_XY['orig']['Y_dev'], pred_all_d[TTA_suffix])), index=False)        
      print(f'run{run} {TTA_suffix} ' + "{:d}folds {:.5f}".format(n_fold, log_loss(d_XY['orig']['Y_dev'], pred_all_d[TTA_suffix]))) # Calculate the log loss 


