'''
 
    usage:
        python anonymyie_dataset.py input.tsv [output_folder]

    given a dataset generates four anonymized datasets saving them into four different output files.
    output folder defaults to .

'''

import pandas as pd
import sys
from tqdm import tqdm
import os

def generate_aug_files(input_tsv, output_folder):

    output_base_fname = os.path.basename(input_tsv).replace ('.tsv', '')

    for names in tqdm ([ { 'female':['Alice','Kate'], 'male': ['John','Michael']},
                { 'female':['Elizabeth','Mary'], 'male': ['James','Henry']},
                { 'female':['Kate','Elizabeth'], 'male': ['Michael','James']},
                { 'female':['Mary','Alice'], 'male': ['Henry','John']}], desc="datasets"):

        output_fname = output_folder + '/' + output_base_fname + '_' + '_'.join([names['female'][0],names['female'][1],names['male'][0],names['male'][1]]) +'.tsv'

        df = pd.read_csv(input_tsv, sep="\t")

        for i in tqdm(range(df.shape[0]), desc="sentences"):
                
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
        for i in tqdm(range(df.shape[0]), desc="sanity check"):        
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

if __name__ == "__main__":
    
    output_folder = "."
    input_tsv = None

    if len (sys.argv) < 2:
        sys.exit ("usage: python anonymyie_dataset.py input.tsv [output_folder]")
    
    input_tsv = sys.argv[1]
    if len (sys.argv) > 2:
        output_folder = sys.argv[2]

    generate_aug_files(input_tsv, output_folder) 