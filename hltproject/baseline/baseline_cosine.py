import sys
from scipy.spatial.distance import cosine
import tqdm

from dataset_utils.parsing import parse_embeddings_dataset

_THRESHOLD = 0.1

def compute_predictions ( train_fname, validation_fname, test_fname ):
    with open(train_fname) as fin, open (train_fname+".cosine.submission", "w") as fout:
        fout.write ("ID,A,B,NEITHER\n")
        for sent in tqdm.tqdm (parse_embeddings_dataset (fin)):
            vecA = sent.embeddings[sent.A_tok_off]
            vecB = sent.embeddings[sent.B_tok_off]
            vec_pron = sent.embeddings[sent.pron_tok_off]

            
#            score_A = ( 2 - cosine ( vecA, vec_pron ) ) / 2
#            score_B = ( 2 - cosine ( vecB, vec_pron ) ) / 2
            score_A = 1 - cosine ( vecA, vec_pron ) 
            score_B = 1 - cosine ( vecB, vec_pron )


#            print(cosine ( vecA, vec_pron ), score_A)
#            print(cosine ( vecB, vec_pron ), score_B)
#            input()

            if score_A + score_B > 0:
                prob_N = min ( 1-score_A, 1-score_B )
                prob_A = score_A * ( 1-prob_N ) / ( score_A + score_B )
                prob_B = 1 - prob_A - prob_N
            else:
                prob_N = 1
                prob_B = 0
                prob_A = 0
            
            print (",".join ([sent.id, str(prob_A), str(prob_B), str(prob_N)]), file=fout)


if __name__ == "__main__":
    compute_predictions ( sys.argv[1], sys.argv[1], sys.argv[1] )
