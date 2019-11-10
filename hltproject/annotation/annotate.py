
import collections
import sys
import os
from getch import getch
from tqdm import tqdm

SentenceWithoutLabel = collections.namedtuple ('SentenceWithoutLabel', ['id', 'text', 'pron', 'pron_off', 'A', 'A_off', 'B', 'B_off', 'url'])

def parse_input_dataset ( fin, lines_to_skip ):
    for i in range ( lines_to_skip): #skip first lines
        next (fin)
    for line in fin:
        id, text, pron, pron_off, A, A_off, B, B_off, url = line.strip().split ('\t')
        A_off, B_off, pron_off = int (A_off), int (B_off), int (pron_off)
        yield SentenceWithoutLabel (id, text, pron, pron_off, A, A_off, B, B_off, url)

def count_lines ( fname ):
    c = 0
    with open(fname) as fin:
        c = sum ( 1 for _ in fin )
    return c

def annotate_sentence ( sent ):
    text_to_print = sent.text
    A_off, B_off, P_off = sent.A_off, sent.B_off, sent.pron_off
    for i in range (3):
        offset = max ( A_off, B_off, P_off )
        if offset == A_off:
            text_to_print = text_to_print[0:A_off] + "\033[94m(A)" + sent.A + "\033[0m" + text_to_print[A_off+len(sent.A):]
            A_off = -1
        if offset == B_off:
            text_to_print = text_to_print[0:B_off] + "\033[91m(B)" + sent.B + "\033[0m" + text_to_print[B_off+len(sent.B):]
            B_off = -1
        if offset == P_off:
            text_to_print = text_to_print[0:P_off] + "\033[92m" + sent.pron + "\033[0m" + text_to_print[P_off+len(sent.pron):]
            P_off = -1
    
    print()
    print (sent.id)
    print (text_to_print)

    print()
    print ("\033[94mA:" + sent.A + "\033[0m" + " " + "\033[91mB:" + sent.B + "\033[0m" + " " + "\033[92mP:" + sent.pron + "\033[0m")
    
    ch = ""
    while not ch.lower() in ['a', 'b', 'n']:
        ch = getch()
    if ch.lower() == 'a':
        return "TRUE", "FALSE"
    if ch.lower() == 'b':
        return "FALSE", "TRUE"
    return "FALSE", "FALSE"

def main ( fname ):
    output_fname = fname.replace(".tsv", "") + "_with_labels.tsv"
    lines_to_skip = 1
    fout = None
    if os.path.exists ( output_fname ):
        lines_to_skip = count_lines ( output_fname )
        fout = open (output_fname, "a")
        print ("output file exists: resuming from line {}.".format(lines_to_skip))
    else:
        lines_to_skip = 1
        fout = open (output_fname, "w")
        fout.write ("ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\tURL\n")
        print ("creating output file and starting from scratch.")
        
    fin = open (fname)
    for sent in tqdm(parse_input_dataset (fin, lines_to_skip)):
        A_coref, B_coref = annotate_sentence (sent)
        fout.write ("\t".join ([sent.id, sent.text, sent.pron, str(sent.pron_off), sent.A, str(sent.A_off), A_coref, sent.B, str(sent.B_off), B_coref, sent.url]) + "\n" )
        os.system ("clear")
    
if __name__ == "__main__":
    main ( sys.argv[1] )
