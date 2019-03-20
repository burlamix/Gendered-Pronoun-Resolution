import numpy as np
import collections

Sentence = collections.namedtuple ('Sentence', ['id', 'tokens', 'embeddings', 'A_tok_off', 'A_coref', 'B_tok_off', 'B_coref', 'pron_tok_off'])

def str_to_bool ( s ):
    return True if s.lower() == 'true' else False

def str_to_embedding ( s ):
    return np.array ( [float(x) for x in s.split()] )

def parse_embeddings_dataset ( fin ):
    next (fin) #skip first line
    first = True
    for line in fin:
        #print ("line---",line[:50])
        if line == '\n':
            sent = Sentence (id, toks, embs, int(Ao), str_to_bool(Ac), int(Bo), str_to_bool(Bc), int (po))
            #print ("returining", sent)
            #input ()
            first = True
            yield sent
        elif first:
            first = False
            id, po, Ao, Ac, Bo, Bc = line.split ('\t')
            toks = []
            embs = []
        else:
            str_tok, str_emb = line.split('\t')
            toks.append (str_tok)
            embs.append (str_to_embedding (str_emb))
            
