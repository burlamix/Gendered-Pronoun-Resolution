
import logging
import logging.config
import hltproject.utils.config as cutils

from shutil import copyfile

logging.config.dictConfig(
    cutils.load_logger_config_file())
logger = logging.getLogger ( __name__ )
logger.setLevel (logging.INFO)


def original_notebook_preprocessing ( is_inference, path, input_tsv_fname ):
    '''
    param is_inference True for inference; False for training
    param path folder where to save results
    param input_tsv_fname input filename to preprocess
    '''

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
    import subprocess
    import urllib

    import spacy

    logger.info ("model_7.original_notebook_preprocessing input_tsv_filename {} is_inference {}".format(input_tsv_fname, is_inference))
    
    #PATH CONSTANTS FOR CONSISTENCY WITH OTHER NOTEBOOKS 
    AUGMENTED_FILES_PREFIX = path + "/input/" + os.path.basename (input_tsv_fname).split('.')[0]
    EMBEDDINGS_FILES_PREFIX = path + "/embeddings/" + os.path.basename(input_tsv_fname).split('.')[0] + '_'
    LINGUI_CSV_FNAME = path + "/output/" + os.path.basename (input_tsv_fname).split('.')[0] + "_lingui_df.csv"
    DIST_CSV_FNAME = path + '/output/'+os.path.basename (input_tsv_fname).split('.')[0]+'_dist_df.csv'
    EXTRACT_FEATURES_PATH = "model_7/extract_features.py"
    
    logger.info ("copying input file into input/ folder")
    copyfile ( input_tsv_fname, path + "/input/" + os.path.basename (input_tsv_fname) )

    logger.info ("loading spacy extensions")
    try:
        nlp = spacy.load('en_core_web_lg')
    except:
        print ("FIXME 2019-08-24 \n\
               This class requires the en_core_web_lg model for the spacy library to be downloaded. \n\
               Apparently there is no way to download it internally from spacy. \n\
               Please run the command \n\
                   python -m spacy download --user en_core_web_lg")
        sys.exit(1)

    #downloading weights and cofiguration file for the model
    def get_bert_model(CASED, LARGE):

        if CASED and LARGE:           model_name = 'cased_L-24_H-1024_A-16'
        elif not CASED and LARGE:     model_name = 'uncased_L-24_H-1024_A-16'
        elif CASED and not LARGE:     model_name = 'cased_L-12_H-768_A-12'
        elif not CASED and not LARGE: model_name = 'uncased_L-12_H-768_A-12'

        local_file_name = path + "/" + model_name + ".zip"
        remote_file_name = 'https://storage.googleapis.com/bert_models/2018_10_18/'+ model_name + ".zip"
        if not os.path.exists(local_file_name):

            urllib.request.urlretrieve(remote_file_name, local_file_name)        
            with zipfile.ZipFile(local_file_name,"r") as zip_ref:
                zip_ref.extractall(path)
        
    logger.info ("getting bert models")

    get_bert_model(CASED = False, LARGE = True)    
    get_bert_model(CASED = True,  LARGE = True)    
    
    # import modeling
    # import extract_features
    # import tokenization
    # import tensorflow as tf

    def compute_offset_no_spaces(text, offset):
        count = 0
        for pos in range(offset):
            if text[pos] != " ": count +=1
        return count

    # def count_chars_no_special(text):
    #     if text=='#': return 1  
    #     count = 0
    #     special_char_list = ["#"]
    #     for pos in range(len(text)):
    #         if text[pos] not in special_char_list: count +=1
    #     return count

    def count_length_no_special(text):
        if text=='#': return 1
        count = 0
        special_char_list = ["#", " "]
        for pos in range(len(text)):
            if text[pos] not in special_char_list: count +=1
        return count

    # for each input tsv file, generate 4 augmented tsv files and save to Drive
    def generate_aug_files(input_tsv, output_prefix):

        for names in [ { 'female':['Alice','Kate'], 'male': ['John','Michael']},
                        { 'female':['Elizabeth','Mary'], 'male': ['James','Henry']},
                        { 'female':['Kate','Elizabeth'], 'male': ['Michael','James']},
                        { 'female':['Mary','Alice'], 'male': ['Henry','John']}]:
            logger.info ("name set: {}".format(names))
            output_fname = output_prefix + '_' + '_'.join([names['female'][0],names['female'][1],names['male'][0],names['male'][1]]) +'.tsv'
            if os.path.exists (output_fname):
                continue
    
            df = pd.read_csv(input_tsv, sep="\t")
        
            for i in tqdm(range(df.shape[0])):
                    
                do_A = True
                do_B = True
            
                text = df.loc[i].Text
                A = df.loc[i].A
                B = df.loc[i].B
                P = df.loc[i].Pronoun
            
                if A in B or B in A:
                    continue
                
                gender = 'female' if df.loc[i,'Pronoun'].lower() in ['her','she'] else 'male'        
                
                ## if placeholder names appear in text, skip to avoid confusion
                if names[gender][0] in text or names[gender][1] in text:
                    continue
        
                # There are names like "M"
                if A in names[gender][0] or B in names[gender][0] or\
                A in names[gender][1] or B in names[gender][1]:
                    continue
                
                ## if name too long, skip 
                if len(A.split(' ')) > 2: do_A=False
                if len(B.split(' ')) > 2: do_B=False
                
                ## if either first or last name appearn with full name, skip
                if len(A.split(' '))==2:
                    if text.count(A.split(' ')[0]) > text.count(A) or text.count(A.split(' ')[1]) > text.count(A):
                        do_A = False
                if len(B.split(' '))==2:
                    if text.count(B.split(' ')[0]) > text.count(B) or text.count(B.split(' ')[1]) > text.count(B):
                        do_B = False
                if not do_A and not do_B: continue       
            
                            
                Aoff = df.loc[i,'A-offset']
                Boff = df.loc[i,'B-offset']
                Poff = df.loc[i,'Pronoun-offset']    
        
                if do_A:    
                    while(A in text):
                        Apos = text.index(A)    
                        text = text.replace(A,names[gender][0] ,1)
                        if Apos < Aoff: Aoff += len(names[gender][0])-len(A)        
                        if Apos < Boff: Boff += len(names[gender][0])-len(A)
                        if Apos < Poff: Poff += len(names[gender][0])-len(A)
                    df.loc[i,'A'] = names[gender][0]                    
        
                if do_B:        
                    while(B in text):
                        Bpos = text.index(B)    
                        text = text.replace(B,names[gender][1] ,1)
                        if Bpos < Poff: Poff += len(names[gender][1])-len(B)
                        if Bpos < Boff: Boff += len(names[gender][1])-len(B)        
                        if Bpos < Aoff: Aoff += len(names[gender][1])-len(B)        
                    df.loc[i,'B'] = names[gender][1] 
            
                df.loc[i,'A-offset'] = Aoff
                df.loc[i,'B-offset'] = Boff
                df.loc[i,'Pronoun-offset'] = Poff
                df.loc[i,'Text'] = text

            # sanity check
            for i in tqdm(range(df.shape[0])):        
                text = df.loc[i].Text
                A = df.loc[i].A
                B = df.loc[i].B
                P = df.loc[i].Pronoun
                Aoff = df.loc[i]['A-offset']
                Boff = df.loc[i]['B-offset']
                Poff = df.loc[i]['Pronoun-offset'] 
                assert text[Aoff:(Aoff+len(A))]==A
                assert text[Boff:(Boff+len(B))]==B
                assert text[Poff:(Poff+len(P))]==P   
                        
            df.to_csv(output_fname, sep="\t",index=False) 
    
    logger.info ("generating aug files")
    generate_aug_files(input_tsv_fname, AUGMENTED_FILES_PREFIX)    

    ## adjust long text (cut off first few sentences so that last of A/B/Pronoun's offset < 900)
    def adjust_long_text(df_in):
    
        df = df_in.copy()
        try:    
            long_text_idx = df[df.apply(lambda x:max(x['A-offset'],x['B-offset'],x['Pronoun-offset']),axis=1) >= 1000].index.values

            for i in long_text_idx:    
                text = df.loc[i].Text
                doc = nlp(text)

                num_char_left = 800 if text.count('``')>5 else 1000

                # number of char to be cut
                num_char_cut = df.apply(lambda x:max(x['A-offset'],x['B-offset'],x['Pronoun-offset']),axis=1).loc[i] - num_char_left

                A = df.loc[i].A
                B = df.loc[i].B
                P = df.loc[i].Pronoun     

                for _,sent in enumerate(list(doc.sents)):
                    if sent.text in text:
                        sent_start_idx = text.index(sent.text)
                    if sent_start_idx > num_char_cut:
                        break

                if sent_start_idx >= min(df.loc[i,'A-offset'],df.loc[i,'B-offset'],df.loc[i,'Pronoun-offset']):
                    continue

                text = text[sent_start_idx:]
                Aoff = df.loc[i,'A-offset']-sent_start_idx
                Boff = df.loc[i,'B-offset']-sent_start_idx
                Poff = df.loc[i,'Pronoun-offset']-sent_start_idx

                assert text[Aoff:(Aoff+len(A))]==A
                assert text[Boff:(Boff+len(B))]==B
                assert text[Poff:(Poff+len(P))]==P

                df.loc[i,'A-offset'] = Aoff
                df.loc[i,'B-offset'] = Boff
                df.loc[i,'Pronoun-offset'] = Poff
                df.loc[i,'Text'] = text

                print(f'adjusted index {i}')
                print(Aoff, Boff, Poff, len(text))
        except:
            pass

        return df


    def run_bert(data, layer="-2", LARGE=False,CASED=False, MAX_SEQ_LEN = 256, debug=False, is_inference=False):
        '''
        Runs a forward propagation of BERT on input text, extracting contextual word embeddings
        Input: data, a pandas DataFrame containing the information in one of the GAP files

        Output: emb, a pandas DataFrame containing contextual embeddings for the words A, B and Pronoun. Each embedding is a numpy array of shape (768)
        columns: "emb_A": the embedding for word A
                "emb_B": the embedding for word B
                "emb_P": the embedding for the pronoun
                "label": the answer to the coreference problem: "A", "B" or "NEITHER"
        '''
        # From the current file, take the text only, and write it in a file which will be passed to BERT
        
        BS = 8    
        if not CASED and not LARGE:
            bert_zip_name = 'uncased_L-12_H-768_A-12'
            SIZE = 768
        elif CASED and not LARGE:
            bert_zip_name = 'cased_L-12_H-768_A-12'
            SIZE = 768
        elif LARGE and not CASED:
            bert_zip_name = 'uncased_L-24_H-1024_A-16'
            SIZE = 1024
            BS=4
        elif LARGE and CASED:
            bert_zip_name = 'cased_L-24_H-1024_A-16'
            SIZE = 1024
            BS=4    
        bert_zip_name = path + '/' + bert_zip_name
        
        text = data["Text"]
        text.to_csv("input.txt", index = False, header = False)

        # The script extract_features.py runs forward propagation through BERT, and writes the output in the file output.jsonl
        command = "python3.6 " + EXTRACT_FEATURES_PATH + "\
            --input_file=input.txt \
            --output_file=output.jsonl \
            --vocab_file="+bert_zip_name+"/vocab.txt \
            --bert_config_file="+bert_zip_name+"/bert_config.json \
            --init_checkpoint="+bert_zip_name+"/bert_model.ckpt \
            --layers=" + layer + " \
            --max_seq_length=" + str(MAX_SEQ_LEN)+ " \
            --batch_size=" + str(BS)
        if CASED:
            command += ' --do_lower_case=False'

        print(command)
        os.system(command)

        bert_output = pd.read_json("output.jsonl", lines = True)

        os.system("rm output.jsonl")
        os.system("rm input.txt")

        data.index = range(data.shape[0])
    
        index = data.index
        columns = ["emb_A", "emb_B", "emb_P", "label"]
        emb = pd.DataFrame(index = index, columns = columns)
        emb.index.name = "ID"

        for i in range(len(data)): # For each line in the data file
            # get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
            if CASED:
                P = data.loc[i,"Pronoun"]
                A = data.loc[i,"A"]
                B = data.loc[i,"B"]      
            else:
                P = data.loc[i,"Pronoun"].lower()
                A = data.loc[i,"A"].lower()
                B = data.loc[i,"B"].lower()

            # For each word, find the offset not counting spaces. This is necessary for comparison with the output of BERT
            P_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"Pronoun-offset"])
            A_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"A-offset"])
            B_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"B-offset"])
            # Figure out the length of A, B, not counting spaces or special characters
            A_length = count_length_no_special(A)
            B_length = count_length_no_special(B)

            # Initialize embeddings with zeros
            emb_A = np.zeros(SIZE)
            emb_B = np.zeros(SIZE)
            emb_P = np.zeros(SIZE)

            # Initialize counts
            count_chars = 0
            cnt_A, cnt_B, cnt_P = 0, 0, 0
            token_A, token_B, token_P = '','',''

            features = pd.DataFrame(bert_output.loc[i,"features"]) # Get the BERT embeddings for the current line in the data file
            start_idx = 2 if features.loc[1,"token"] == '"' else 1

            for j in range(start_idx,len(features)):  # Iterate over the BERT tokens for the current line; we skip over the first 2 tokens, which don't correspond to words
                token = features.loc[j,"token"]

                # See if the character count until the current token matches the offset of any of the 3 target words
                if count_chars  == P_offset: 
                    token_P += token.replace('#','')
                    # print(token)
                    emb_P += np.array(features.loc[j,"layers"][0]['values'])
                    cnt_P += 1
                if count_chars in range(A_offset, A_offset + A_length): 
                    token_A += token.replace('#','')    
                    # print(token)
                    emb_A += np.array(features.loc[j,"layers"][0]['values'])
                    cnt_A +=1  
                if count_chars in range(B_offset, B_offset + B_length): 
                    token_B += token.replace('#','')    
                    # print(token)
                    emb_B += np.array(features.loc[j,"layers"][0]['values'])    
                    cnt_B +=1				   
                    # Update the character count
                count_chars += count_length_no_special(token)
            if not (token_A==A.replace(' ','') and token_B==B.replace(' ','') and token_P==P):
                print("assert failed for {:d}".format(i))
                print(token_A,A.replace(' ','') , token_B,B.replace(' ','') , token_P,P)
            # Taking the average between tokens in the span of A or B, so divide the current value by the count	
            emb_A /= cnt_A
            emb_B /= cnt_B
        
            if is_inference:
                label = ''
            else:
                # Work out the label of the current piece of text
                label = "Neither"
                if (data.loc[i,"A-coref"] == True):
                    label = "A"
                if (data.loc[i,"B-coref"] == True):
                    label = "B"

            # Put everything together in emb
            emb.iloc[i] = [emb_A, emb_B, emb_P, label]
        
        return emb

    ### extract original features and 4 TTA features
    LARGE = True


    def extract_data(input_tsv_path, output_json_path, start_idx=None, end_idx=None, is_inference=False):
        logger.info ("extracting data from input file {}, output {}".format(input_tsv_path, output_json_path))
        if os.path.exists(output_json_path):
            logger.info ("already computed, skipping")
            return
        data = pd.read_csv(input_tsv_path, sep = '\t')
        if start_idx!=None and end_idx!=None:
            data = data.iloc[start_idx:end_idx]
        data = adjust_long_text(data)
        emb = run_bert(data, LARGE=LARGE, CASED=CASED,layer=layer, MAX_SEQ_LEN=MAX_SEQ_LEN, is_inference=is_inference)
        emb.to_json(output_json_path, orient = 'columns')   
        gc.collect()


    logger.info ("Running bert")
    for CASED in [True, False]:  
        logger.info ("CASED {}".format(CASED))
        for layer in ["-3","-4"]:
            logger.info ("layer {}".format(layer))

            MAX_SEQ_LEN = 256

            suffix = ('_'+ str(MAX_SEQ_LEN)) if MAX_SEQ_LEN != 256 else ""
            if CASED: suffix += '_CASED'
            if LARGE: suffix += '_LARGE'

            TTA_suffixes = [ '',
                            'Alice_Kate_John_Michael',
                            'Elizabeth_Mary_James_Henry',
                            'Kate_Elizabeth_Michael_James',
                            'Mary_Alice_Henry_John']

            for TTA_suffix in tqdm(TTA_suffixes):

                if TTA_suffix:
                    aug_fname = AUGMENTED_FILES_PREFIX+'_'+TTA_suffix+'.tsv'
                else:
                    aug_fname = AUGMENTED_FILES_PREFIX+'.tsv'

                if is_inference:
                
                    num_test = pd.read_csv(input_tsv_fname,sep='\t').shape[0]
                    n_chunk = int(np.ceil(num_test/1000))
                    for i in range(n_chunk):
                        print(f"chunk{i}")          
                        extract_data(aug_fname, 
                                    EMBEDDINGS_FILES_PREFIX + layer+ suffix +TTA_suffix+ f'_{i}_fix_long_text.json',
                                    start_idx = i*1000, end_idx = min(num_test,(i+1)*1000), is_inference = is_inference)  
                else:
                    extract_data(aug_fname, EMBEDDINGS_FILES_PREFIX + layer+ suffix + '_' + TTA_suffix+ '_fix_long_text.json', is_inference = is_inference)



    def bs(lens, target):
        low, high = 0, len(lens) - 1

        while low < high:
            mid = low + int((high - low) / 2)

            if target > lens[mid]:
                low = mid + 1
            elif target < lens[mid]:
                high = mid
            else:
                return mid + 1

        return low

    def bin_distance(dist):
        
        buckets = [1, 2, 3, 4, 5, 8, 16, 32, 64]  
        low, high = 0, len(buckets)
        while low < high:
            mid = low + int((high-low) / 2)
            if dist > buckets[mid]:
                low = mid + 1
            elif dist < buckets[mid]:
                high = mid
            else:
                return mid

        return low

    def distance_features(P, A, B, char_offsetP, char_offsetA, char_offsetB, text, URL):
        
        doc = nlp(text)
        
        lens = [token.idx for token in doc]
        mention_offsetP = bs(lens, char_offsetP) - 1
        mention_offsetA = bs(lens, char_offsetA) - 1
        mention_offsetB = bs(lens, char_offsetB) - 1
        
        mention_distA = mention_offsetP - mention_offsetA 
        mention_distB = mention_offsetP - mention_offsetB
        
        splited_A = A.split()[0].replace("*", "")
        splited_B = B.split()[0].replace("*", "")
        
        if re.search(splited_A[0], str(URL)):
            contains = 0
        elif re.search(splited_B[0], str(URL)):
            contains = 1
        else:
            contains = 2
        
        dist_binA = bin_distance(mention_distA)
        dist_binB = bin_distance(mention_distB)
        output =  [dist_binA, dist_binB, contains]
        
        return output

    def extract_dist_features(df):
        
        index = df.index
        columns = ["D_PA", "D_PB", "IN_URL"]
        dist_df = pd.DataFrame(index = index, columns = columns)

        for i in tqdm(range(len(df))):
            
            text = df.loc[i, 'Text']
            P_offset = df.loc[i,'Pronoun-offset']
            A_offset = df.loc[i, 'A-offset']
            B_offset = df.loc[i, 'B-offset']
            P, A, B  = df.loc[i,'Pronoun'], df.loc[i, 'A'], df.loc[i, 'B']
            URL = df.loc[i, 'URL']
            
            dist_df.iloc[i] = distance_features(P, A, B, P_offset, A_offset, B_offset, text, URL)
            
        return dist_df

    logger.info ("extracting dist features")
    if is_inference:
        stage2_df = pd.read_csv(input_tsv_fname,sep='\t')
        stage2_dist_df = extract_dist_features(stage2_df)
        out_csv_path = DIST_CSV_FNAME
        if os.path.exists(out_csv_path): os.remove(out_csv_path)
        stage2_dist_df.to_csv(out_csv_path, index=False)
    
    else:
        inp_df  = pd.read_csv(input_tsv_fname,sep='\t')
        dist_df  = extract_dist_features(inp_df)
        dist_df.to_csv(DIST_CSV_FNAME, index=False)
  
    # Two useful syntactic relations
    def domain(t):
        while not t._.subj and not t._.poss and\
                not (t.dep_ == 'xcomp' and t.head._.obj) and\
                t != t.head:
            t = t.head
        return t

    def ccom(t):
        return [t2 for t2 in t.head._.d]

    spacy.tokens.doc.Doc.set_extension(
        'to', method=lambda doc, offset: [t for t in doc if t.idx == offset][0], force=True)
    spacy.tokens.token.Token.set_extension(
        'c', getter=lambda t: [c for c in t.children], force=True)
    spacy.tokens.token.Token.set_extension(
        'd', getter=lambda t: [c for c in t.sent if t in list(c.ancestors)], force=True)
    spacy.tokens.token.Token.set_extension(
        'subj', getter=lambda t: ([c for c in t._.c if c.dep_.startswith('nsubj')] + [False])[0], force=True)
    spacy.tokens.token.Token.set_extension(
        'obj', getter=lambda t: ([c for c in t._.c if c.dep_.startswith('dobj')] + [False])[0], force=True)
    spacy.tokens.token.Token.set_extension(
        'poss', getter=lambda t: ([c for c in t._.c if c.dep_.startswith('poss')] + [False])[0], force=True)
    spacy.tokens.token.Token.set_extension(
        'span', method=lambda t, t2: t.doc[t.i:t2.i] if t.i < t2.i else t.doc[t2.i:t.i], force=True)
    spacy.tokens.token.Token.set_extension('domain', getter=domain, force=True)
    spacy.tokens.token.Token.set_extension('ccom', getter=ccom, force=True)

    # Disqualification functions

    # Prune candidate list given a disqualifying condition (a set of tokens)
    def applyDisq(condition, candidates, candidate_dict, debug = False):
        badnames = sum([nameset(c, candidate_dict) for c in candidates if c in condition[0]], [])
        badcands = [c for c in candidates if c.text in badnames]
        if debug and len(badcands) > 0: print('Disqualified:', badcands, '<', condition[1])
        return [c for c in candidates if c not in badcands]

    # Apply a list of disqualifying conditions
    def applyDisqs(conditions, candidates, candidate_dict, debug = False):
        for condition in conditions:
            if len(candidates) < 1: return candidates
            candidates = applyDisq(condition, candidates, candidate_dict, debug)
        return candidates

    # Pass the list of disqualifying conditions for possessive pronouns (his, her)
    def disqGen(t, candidates, candidate_dict, debug = False):
        conds = [(t._.ccom,
                "disqualify candidates c-commanded by genpn; e.g. e.g. *Julia read his_i book about John_i's life."),
                ([t2 for t2 in candidates if t in t2._.ccom and t2.head.dep_ == 'appos'],
                "disqualify candidates modified by an appositive with genpn; e.g. *I wanted to see John_i, his_i father.")
                ]
        return applyDisqs(conds, candidates, candidate_dict, debug)

    # Pass the list of list of disqualifying conditions for other pronouns
    def disqOthers(t, candidates, candidate_dict, debug = False):
        conds = [([t2 for t2 in t._.ccom if t2.i > t.i],
                "disqualify candidates c-commanded by pn, unless they were preposed;\
                e.g. *He_i cried before John_i laughed. vs. Before John_i laughed, he_i cried."),
                ([t2 for t2 in candidates if t in t2._.ccom and t2._.domain == t._.domain
                and not (t.head.text == 'with' and t.head.head.lemma_ == 'take')],
                "disqualify candidates that c-command pn, unless in different domain;\
                e.g. Mary said that *John_i hit him_i. vs. John_i said that Mary hit him_i;\
                random hard-coded exception: `take with'"),
                ([t2 for t2 in candidates if t2._.domain.dep_ == 'xcomp' and t2._.domain.head._.obj and t2 == t2._.domain.head._.obj],
                "for xcomps with subjects parsed as upstairs dobj, disallow coref with that dobj;\
                e.g. *Mary wanted John_i to forgive him_i.")
                ]
        return applyDisqs(conds, candidates, candidate_dict, debug)

    # Decide whether possessive or not and call appropriate function
    def disq(t, candidates, candidate_dict, debug = False):
        func = disqGen if t.dep_ == 'poss' else disqOthers
        candidates = func(t, candidates, candidate_dict, debug)
        return candidates

    # Name functions

    # Find word of interest at provided offset; sometimes parsed words don't align with provided data, so need to look back
    def find_head(w, wo, doc):
        t = False; backtrack = 0
        while not t:
            try:
                t = doc._.to(wo)
            except IndexError:
                wo -= 1; backtrack += 1
        while t.dep_ == 'compound' and t.head.idx >= wo and t.head.idx < len(w) + wo + backtrack: t = t.head
        return t

    # Returns subsequences of a name
    def subnames(name):
        #if type(name) != str: name = candidate_dict[name]
        parts = name.split(' ')
        subnames_ = []
        for i in range(len(parts)): 
            for j in range(i + 1, len(parts) + 1): 
                sub = ' '.join(parts[i:j])
                if len(sub) > 2: subnames_.append(sub)
        return subnames_

    # Returns subsequences of a name unless potentially ambiguous (if another candidate picks out same subsequence)
    def nameset(name, candidate_dict):
        if type(name) != str: name = candidate_dict[name]
        subnames_ = [sn for sn in subnames(name)]
        return [c for c in subnames_ if c not in sum([subnames(c) for c in candidate_dict.values() 
                                                    if c not in subnames_ and name not in subnames(c)], [])]

    # Given the original candidate dict and the final candidate list, returns new dict grouping putative candidate instances under a single key
    def candInstances(candidates, candidate_dict):
        candidates_by_name = {}
        for c in sorted(candidates, key = lambda c: len(candidate_dict[c]), reverse = True):
            name = candidate_dict[c]
            for name2 in candidates_by_name.keys():
                if name in nameset(name2, candidate_dict): name = name2; break
            candidates_by_name[name] = candidates_by_name.get(name, []) + [c]
        return candidates_by_name

    import gender_guesser.detector as gender 
    gd = gender.Detector()

    # Needed to prune candidate dict-- removes non-provided candidates that don't match in most common gender with pn
    def filterGender(candidates_by_name, a, b, pn):
        badnames = []
        gender = 'female' if pn in ['She', 'she', 'her', 'Her'] else 'male'
        for name in candidates_by_name.keys():
            if a in subnames(name) or b in subnames(name): continue
            genderii = gd.get_gender(name.split(' ')[0])
            if gender == 'male' and genderii == 'female': badnames += [name]; continue
            if gender == 'female' and genderii == 'male': badnames += [name]; continue
        for name in badnames: candidates_by_name.pop(name)
        return candidates_by_name

    # Metrics

    from urllib.parse import unquote
    import re

    # Authors' metric 1: Does the Wikipedia url contain the candidate's name?
    def urlMatch(a, b, url, candidate_dict):
        url = re.sub('[^\x00-\x7F]', '*', unquote(url.split('/')[-1])).replace('_', ' ').lower()
        return {'a_url': (sorted([len(n.split(' ')) for n in nameset(a.lower(), candidate_dict) if n in nameset(url, candidate_dict)], reverse = True) + [0])[0],
                'b_url': (sorted([len(n.split(' ')) for n in nameset(b.lower(), candidate_dict) if n in nameset(url, candidate_dict)], reverse = True) + [0])[0]}

    # Authors' metric 2: When pn is subject or object, does the candidate match?
    def parallel(t1, t2):
        if t1.dep_.startswith('nsubj'): return t2.dep_.startswith('nsubj')
        if t1.dep_.startswith('dobj'): return t2.dep_.startswith('dobj')
        if t1.dep_.startswith('dative'): return t2.dep_.startswith('dative')
        return False

    # Depth from a node to a parent node
    def depthTo(t1, t2):
        depth = 0
        while t1 != t2 and t1 != t1.head:
            t1 = t1.head
            depth += 1
        return depth

    # Syntactic distance within a single tree
    def nodeDist(t1, t2):
        if t1 == t2: return 0
        if t2 in t1._.d: return depthTo(t2, t1)
        if t1 in t2._.d: return depthTo(t1, t2)
        t = t1
        while t1 not in t._.d or t2 not in t._.d and t != t.head: t = t.head
        return depthTo(t1, t) + depthTo(t2, t)

    # Authors' metric 3: Syntactic distance (within or across trees)
    def synDist(t, pn, doc, debug = False):
        doc_sents = list(doc.sents)
        sspan = doc_sents.index(pn.sent) - doc_sents.index(t.sent)
        if sspan == 0: # same sentence
            dist = nodeDist(t, pn)
        else: # different sentence
            dist = nodeDist(pn, doc_sents[doc_sents.index(pn.sent)].root) + nodeDist(t, doc_sents[doc_sents.index(t.sent)].root) # dist from two roots
        if debug: 
            print('pn dist:', nodeDist(pn, doc_sents[doc_sents.index(pn.sent)].root), '; t dist:',
                nodeDist(t, doc_sents[doc_sents.index(t.sent)].root), '; span:', sspan)
        sspan = abs(sspan) * 1 if sspan >= 0 else abs(sspan) * 1.3 # less local if not preceding
        return dist + sspan# * 0.7

    # Character distance
    def charDist(t1, t2):
        if t2.idx > t1.idx:
            return t2.idx - t1.idx + len(t1.text)
        else:
            return (t1.idx - t2.idx + len(t2.text)) * 1.3

    # Theta prominence: assign a 0.1 to 1 score based on dep role of candidate -- strong feature
    def thetaProminence(t, mult = 1, debug = False):
        while t.dep_ == 'compound': t = t.head
        if debug: print('t dep_:', t.dep_)
        if t.dep_ == 'pobj': mult = 1.3 if t.head.i < t.head.head.i else 1
        if t._.domain.dep_ == 'advcl': mult = 1.3 if t.head.i < t._.domain.head.i else 1
        if t.dep_.startswith('nsubj'): score = 1
        elif t.dep_.startswith('dobj'): score = 0.8
        elif t.dep_.startswith('dative'): score = 0.6
        elif t.dep_.startswith('pobj'): score = 0.4
        elif t.dep_.startswith('poss'): score = 0.3
        else: score = 0.1
        if debug: print('mult:', mult, '; score:', score)
        return min(1, score * mult)

    # Computes these metrics for each candidate, and returns, for each group of instances (A instances, B instances,\
    # other instances), either the sum, or the highest difference from the mean
    def score(label, candidates_by_name, a_cand, b_cand, func, minsc = None, method = 'sum'):
        if method == 'sum':
            scores = {name: sum([func(t) for t in tokens]) for name, tokens in candidates_by_name.items()}
        elif method == 'meandiff':
            mean = np.mean(sum([[func(t) for t in tokens] for tokens in candidates_by_name.values()], []))
            scores = {name: mean - min([func(t) for t in tokens]) for name, tokens in candidates_by_name.items()}
        sca = scores[a_cand] if a_cand else minsc
        scb = scores[b_cand] if b_cand else minsc
        screst = [v for n, v in scores.items() if n != a_cand and n != b_cand]
        if method == 'sum':
            screst = sum(screst) if len(screst) > 0 else minsc
        elif method == 'meandiff':
            screst = max(screst) if len(screst) > 0 else minsc
        return {'a_' + label: sca, 'b_' + label: scb, 'n_' + label: screst}

    # Load a rowfull of data
    def load_row(data, i):
        return tuple(data.iloc[i])

    # Row by row, populate features
    def annotateSet(data, minsc = None, debug = False, inference=False):
        
        annotated_data = pd.DataFrame() # init placeholder df
        row_batch = []

        for i in tqdm(range(annotated_data.shape[0], data.shape[0])):

            if not inference: id, text, pn, pno, a, ao, ag, b, bo, bg, url = load_row(data, i)        
            if inference: id, text, pn, pno, a, ao, b, bo, url = load_row(data, i)  

            doc = nlp(text) # parse text into doc
            pnt, at, bt = (doc._.to(pno), find_head(a, ao, doc), find_head(b, bo, doc)) # get the tokens that correspond to offsets
            candidate_dict = {e.root: re.sub('\'s$', '', e.text) for e in [e for e in doc.ents if e.root.ent_type_ == 'PERSON']} # first get every PERSON ent as candidate
            candidate_dict.update({c.root: re.sub('\'s$', '', c.text) for c in doc.noun_chunks if c.root.pos_ == 'PROPN' and c.text in sum([subnames(n) for n in candidate_dict.values()], []) and
                                c.root not in candidate_dict.keys()}) # get some missed ones by looking at noun chunks with PROPN roots whose text match part of a candidate but are not already in list
            candidate_dict.update({t: w for t, w in [(at, a), (bt, b)]}) # add provided cands, overwriting in the process

            candidates = disq(pnt, list(candidate_dict.keys()), candidate_dict, debug = False)
            candidates_by_name = candInstances(candidates, candidate_dict)
            candidates_by_name = filterGender(candidates_by_name, a, b, pn)
            a_cand = ([name for name, tokens in candidates_by_name.items() if at in tokens] + [False])[0]
            b_cand = ([name for name, tokens in candidates_by_name.items() if bt in tokens] + [False])[0]

            # init row dict
            if not inference: features = {'id': id, 'label': 0 if ag else 1 if bg else 2}
            if inference: features = {'id': id, 'label': 0 }
            # eliminated or not
            features.update({'a_out': 0 if a_cand else 1, 'b_out': 0 if b_cand else 1})
            # url match or not
            features.update(urlMatch(a, b, url, candidate_dict))
            # c-command or not
            features.update({'a_cc': 1 if a_cand and pnt in at._.ccom else 0, 'b_cc': 1 if b_cand and pnt in bt._.ccom else 0})
            # parallelism score
            features.update(score('par', candidates_by_name, a_cand, b_cand, lambda t: parallel(t, pnt), minsc = minsc))
            # theta prominence score
            features.update(score('th', candidates_by_name, a_cand, b_cand, thetaProminence, minsc = minsc))
            # syntactic distance score
            features.update(score('loc', candidates_by_name, a_cand, b_cand, lambda t: synDist(t, pnt, doc), method='meandiff', minsc = minsc))
            # number of candidates left
            features.update({'n_cands': len(candidates_by_name)})
            # char dist
            features.update(score('cloc', candidates_by_name, a_cand, b_cand, lambda t: charDist(t, pnt), method='meandiff', minsc = minsc))

            row_batch += [features]


        # add rows to placeholder df
        if annotated_data.shape[0] != data.shape[0]: annotated_data = annotated_data.append(row_batch, ignore_index = True)
        
        return annotated_data

    ## post-process: fill na and standardize

    def post_process(df):
    
        cols = ['a_cc', 'a_loc', 'a_out', 'a_th', 'a_url', 'b_cc', 'b_loc', 'b_out', 'b_th', 'b_url']
        df = df[cols]

        df.a_loc.fillna(-5, inplace=True)
        df.b_loc.fillna(-5, inplace=True)

        df.a_th.fillna(0, inplace=True)
        df.b_th.fillna(0, inplace=True)

        df.a_url /=5 
        df.b_url /=5 

        df.a_th /=5
        df.b_th /=5

        df.a_loc += 8
        df.a_loc /= 25
        df.b_loc += 8
        df.b_loc /=25
            
        return df.copy()


    logger.info ("Extracting linguistic features")
    # %%time
    if is_inference:
        stage2_df = pd.read_csv(input_tsv_fname,sep='\t')
        
        ## fixed an input data error in row 1566 where "(t)he" is incorrectly tagged as pronoun
        ##   this is discussed and allowed in https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/89830
        text = stage2_df.loc[1566,'Text']  
        stage2_df.loc[1566,'Text'] = text = text[:313] + ' ' + text[313:]
        stage2_df.loc[1566,'Pronoun-offset'] += 1
        
        stage2_lingui_df = annotateSet(stage2_df, inference=True)
        stage2_lingui_df = post_process(stage2_lingui_df)
        stage2_lingui_df.to_csv(LINGUI_CSV_FNAME, index=False)
        
    else:
        inp_df  = pd.read_csv(input_tsv_fname,sep='\t')
        
        lingui_df  = annotateSet(inp_df)
        lingui_df  = post_process(lingui_df)
        
        lingui_df.to_csv(LINGUI_CSV_FNAME, index=False)

    logger.info ("Done preprocessing {}".format (input_tsv_fname))

