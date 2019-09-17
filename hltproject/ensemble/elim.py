import pickle
import numpy as np


def max_voting(valutaion_arrays):

    ris= np.asarray(valutaion_arrays)
    new_ris= []

    for xid in range(ris.shape[1]):     # for each x-axis
        #print(ris[:, xid, :])      

        max_abs = 0
        for model_eval in ris[:, xid, :]:
            max_a = np.amax(model_eval)
            index = np.where(model_eval == max_a)

            if(max_a>=max_abs):
                max_abs=max_a
                index_abs= index

            a = [0,0,0]

            a[int(index[0])]=1
            new_ris.append(a)

    return new_ris

with open('array_evalutation.pkl', 'rb') as f:
    x = pickle.load(f)

ris= np.asarray(x)

print(max_voting(x))



