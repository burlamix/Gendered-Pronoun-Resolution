
import sys
import os
from typing import NamedTuple
from getch import getch

class Sentence (NamedTuple):
    id: str
    text: str
    pron: str
    pron_off: int
    A: str
    A_off: int
    A_coref: bool
    B: str
    B_off: int
    B_coref: bool
    url: str

def str_to_bool ( s: str ) -> bool:
    return s.lower() == "true"

def parse_input_dataset ( fin, lines_to_skip ) -> Sentence:
    for i in range ( lines_to_skip): #skip first lines
        next (fin)
    for line in fin:
        id, text, pron, pron_off, A, A_off, A_coref, B, B_off, B_coref, url = line.strip().split ('\t')
        A_off, B_off, pron_off, A_coref, B_coref = int (A_off), int (B_off), int (pron_off), str_to_bool (A_coref), str_to_bool (B_coref)
        yield Sentence (id, text, pron, pron_off, A, A_off, A_coref, B, B_off, B_coref, url)

def show_sentence ( sent: Sentence ):
    text_to_print = sent.text
    A_off, B_off, P_off = sent.A_off, sent.B_off, sent.pron_off
    for i in range (3):
        offset = max ( A_off, B_off, P_off )
        if offset == A_off:
            blink = "\033[4m" if sent.A_coref else ''
            text_to_print = text_to_print[0:A_off] + blink + "\033[94m(A)" + sent.A + "\033[0m" + text_to_print[A_off+len(sent.A):]
            A_off = -1
        if offset == B_off:
            blink = "\033[4m" if sent.B_coref else ''
            text_to_print = text_to_print[0:B_off] + blink + "\033[91m(B)" + sent.B + "\033[0m" + text_to_print[B_off+len(sent.B):]
            B_off = -1
        if offset == P_off:
            text_to_print = text_to_print[0:P_off] + "\033[92m" + sent.pron + "\033[0m" + text_to_print[P_off+len(sent.pron):]
            P_off = -1
    
    print()
    print (sent.id, sent.url.split('/')[-1])
    print ()
    print (text_to_print)

    label = "\033[94mA\033[0m" if sent.A_coref else "\033[91mB\033[0m" if sent.B_coref else "N"

    print()
    print ("\033[94mA:" + sent.A + "\033[0m" + " " + "\033[91mB:" + sent.B + "\033[0m" + " " + "\033[92mP:" + sent.pron + "\033[0m")
    print ("current label: {}".format(label))

def main ( fname: str ):
    
    try:
        while True:
            l = input ("semicolon-separated list of sentence ids to show or EOF (Ctrl+D) to exit: ")
            ids_to_show = l.split (";")
            fin = open (fname)
            for sent in parse_input_dataset (fin, 1):
                if sent.id in ids_to_show:
                    show_sentence (sent)
                    getch ()
                    os.system ("clear")
   
    except EOFError:
        print ("Bye.")
    
 
if __name__ == "__main__":
    main ( sys.argv[1] )
