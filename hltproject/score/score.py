
from hltproject.dataset_utils.parsing import parse_input_dataset, parse_prediction_file

import logging
import logging.config
import hltproject.utils.config as cutils
import collections

import math

logging.config.dictConfig(
    cutils.load_logger_config_file())
logger = logging.getLogger ( __name__ )

Prediction = collections.namedtuple ('Prediction', ['id', 'A_prob', 'B_prob', 'N_prob'])


def parse_pandas ( fin ):

    for index, row in fin.iterrows():
        yield Prediction ( row["ID"],row["A"],row["B"],row["NEITHER"] )


def compute_loss_df ( val_probas_df, input_fname ):
    


    gold_classes = {}
    with  open ( input_fname ) as input_fin:
        for sent in parse_input_dataset ( input_fin ):
            gold_classes[sent.id] = 1 if sent.A_coref else 2 if sent.B_coref else 3
        
        logssum = 0
        N = 0
        for pred in parse_pandas(val_probas_df):
            if not pred.id in gold_classes:
                logger.warn ("unknown prediction id: {}".format(pred.id))
                continue

            #LA PROBABILITà CHE HO SELEZIONATO
            prob = pred[ gold_classes[pred.id] ]
            logssum += math.log ( max ( min ( prob, 1-1e-15 ), 1e-15 ) )
            N += 1
        
        loss = -logssum / N
        
        logger.info ("loss\t{}".format(loss))
        print ("loss\t{}".format(loss))
            

def compute_loss ( model_fname, input_fname, enable_print=True, print_wrong_predictions=False ):
    
    gold_classes = {}
    with open ( model_fname ) as model_fin, open ( input_fname ) as input_fin:
        for sent in parse_input_dataset ( input_fin ):
            gold_classes[sent.id] = 1 if sent.A_coref else 2 if sent.B_coref else 3
        
        errored_ids = []
        logssum = 0
        N = 0
        for pred in parse_prediction_file ( model_fin ):
            if not pred.id in gold_classes:
                logger.warn ("unknown prediction id: {}".format(pred.id))
                continue
            
            prob = pred[ gold_classes[pred.id] ]
            logssum += math.log ( max ( min ( prob, 1-1e-15 ), 1e-15 ) )
            N += 1

            if prob < 0.5:
                errored_ids.append (pred.id)

            # logger.debug ("prediction: %s", pred)
            # logger.debug ("gold class for this prediction: %d", gold_classes[pred.id])
            # logger.debug ("p, clamped: %f", max ( min ( prob, 1-1e-15 ), 1e-15  ))
            # logger.debug ("log (p): %f", math.log (max ( min ( prob, 1-1e-15 ), 1e-15  )))
            # logger.debug ("logsum so far: %f", logssum)
            # input ()

        loss = -logssum / N
        
        if enable_print:
            logger.info ("loss\t{}".format(loss))
            print ("loss\t{}".format(loss))
            if print_wrong_predictions:
                logger.info ("number of wrong predictions\t{}/{}".format(len(errored_ids), N))
                logger.info ("ids of the wrong predicted sentences: {}".format(';'.join(errored_ids)))
    
    return loss
            
