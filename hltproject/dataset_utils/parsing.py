import numpy as np
import collections

RawSentence = collections.namedtuple ('RawSentence', ['id', 'text', 'pron', 'pron_off', 'A', 'A_off', 'A_coref', 'B', 'B_off', 'B_coref', 'url'])
Sentence = collections.namedtuple ('Sentence', ['id', 'tokens', 'embeddings', 'A_tok_off', 'A_coref', 'B_tok_off', 'B_coref', 'pron_tok_off'])
Prediction = collections.namedtuple ('Prediction', ['id', 'A_prob', 'B_prob', 'N_prob'])

def str_to_bool ( s ):
    return s.lower() == 'true'

def str_to_embedding ( s ):
    return np.array ( [float(x) for x in s.split()] )

def parse_input_dataset ( fin ):
    next (fin) #skip first line
    for line in fin:
        id, text, pron, pron_off, A, A_off, A_coref, B, B_off, B_coref, url = line.split ('\t')
        A_off, B_off, pron_off = int (A_off), int (B_off), int (pron_off)
        A_coref, B_coref = str_to_bool (A_coref), str_to_bool (B_coref)
        yield RawSentence (id, text, pron, pron_off, A, A_off, A_coref, B, B_off, B_coref, url)

def parse_embeddings_dataset ( fname ):
    fin = open (fname)
    next (fin) #skip first line
    first = True
    for line in fin:
        line = line.strip ()
        # print ("line---",line[:50])
        if line == '':
            sent = Sentence (id, toks, embs, int(Ao), str_to_bool(Ac), int(Bo), str_to_bool(Bc), int (po))
            # print ("returning", sent)
            # input ()
            first = True
            yield sent
        elif first:
            first = False
            print("\n\n")
            print(line)
            # input()
            id, po, Ao, Ac, Bo, Bc = line.split ('\t')

            #print(id)
            #print(po)
            #print(Ao)
            #print(Ac)
            #print(Bo)
            #print(Bc)
            

            toks = []
            embs = []
        else:
            str_tok, str_emb = line.split('\t')
            toks.append (str_tok)
            embs.append (str_to_embedding (str_emb))

def parse_prediction_file ( fin ):
    next (fin) # skip first line
    for lineno, line in enumerate (fin, 1):
        try:
            id, sa, sb, sn = line.split(',')
            total = float(sa) + float(sb) + float(sn)
            pa = float(sa)/total
            pb = float(sb)/total
            pn = float(sn)/total
            yield Prediction ( id, pa, pb, pn )
        except:
            raise ValueError('Wrong prediction file format.\nFile: {}\nline number: {}\nline: {}'.format(fin, lineno, line))
