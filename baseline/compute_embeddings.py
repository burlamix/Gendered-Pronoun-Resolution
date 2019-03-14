'''
 usage python3 compute_embeddings.py filename.tsv
 
 computes elmo embeddings for a GAP coreference dataset.
 
 Input file must have one line, 11 columns per sentence: ID Text Pronoun Pronoun-offset A A-offset A-coref B B-offset B-coref URL
 
 Output files have one block of lines per sentence:
  - the first line has 6 columns: ID Pronoun-tok-offset A-tok-offset A-coref B-tok-offset B-coref
  - one line per token in the text follows. Each line has two columns: token embedding
  - one blank line follow
 
 One output file is produced for each layer of the elmo network
 
'''
from allennlp.commands.elmo import ElmoEmbedder
import scipy
import sys
import re
import tqdm

def compute_embeddings ( fname ):
    
    elmo = ElmoEmbedder()
    
    fouts = []
    for i in (0,1,2):
        fout = open ( fname+".embeddings.l"+str(i), "w")
        fout.write ( '\t'.join(("ID", "Pronoun-tok-offset", "A-tok-offset", "A-coref", "B-tok-offset", "B-coref"))+'\n' )
        fouts.append (fout)
		
    with open (fname) as fin:
        next(fin) # discard first line
        for line in tqdm.tqdm(fin):
            id, text, _, pron_off, A, A_off, A_coref, B, B_off, B_coref, _ = line.split ('\t')
            
            A_token = A.replace (" ", "_")
            B_token = B.replace (" ", "_")

            text = text.replace (A, A_token).replace (B, B_token)
            
            A_tok_off = re.subn ('\\s', '', text[:int(A_off)])[1]
            B_tok_off = re.subn ('\\s', '', text[:int(B_off)])[1]
            pron_tok_off = re.subn ('\\s', '', text[:int(pron_off)])[1]
            
            tokens = text.split ()
            vectors = elmo.embed_sentence(tokens)
            
                
            for i in (0,1,2):
                fouts[i].write ( '\t'.join((id, str(pron_tok_off), str(A_tok_off), A_coref, str(B_tok_off), B_coref))+'\n' )
                for tok, vec in zip (tokens, vectors[i]):
                    fouts[i].write (tok + '\t' + ' ' .join( str(x) for x in vec )+'\n' )
                fouts[i].write ('\n')


if __name__ == "__main__":
    compute_embeddings ( sys.argv[1] )

