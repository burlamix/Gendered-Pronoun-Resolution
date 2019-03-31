
from hltproject.dataset_utils.parsing import parse_input_dataset, parse_prediction_file

import logging
import logging.config
import hltproject.utils.config as cutils

import math

logging.config.dictConfig(
    cutils.load_logger_config_file())
logger = logging.getLogger ( __name__ )


def compute_loss ( model_fname, input_fname ):
    
    gold_classes = {}
    with open ( model_fname ) as model_fin, open ( input_fname ) as input_fin:
        for sent in parse_input_dataset ( input_fin ):
            gold_classes[sent.id] = 1 if sent.A_coref else 2 if sent.B_coref else 3
        
        logssum = 0
        N = 0
        for pred in parse_prediction_file ( model_fin ):
            if not pred.id in gold_classes:
                logger.warn ("unknown prediction id: {}".format(pred.id))
                continue
            
            prob = pred[ gold_classes[pred.id] ]
            logssum += math.log ( max ( min ( prob, 1-1e-15 ), 1e-15 ) )
            N += 1
        
        loss = -logssum / N
        
        logger.info ("loss\t{}".format(loss))
        print ("loss\t{}".format(loss))
            
