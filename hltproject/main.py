"""Welcome to hltproject.

This is the entry point of the application.
"""

import argparse
import os

import logging
import logging.config
import hltproject.utils.config as cutils

from hltproject.dataset_utils.compute_embeddings import compute_embeddings
from hltproject.dataset_utils.compute_bert_embeddings import compute_bert_embeddings
from hltproject.baseline import baseline_cosine
from hltproject.baseline import baseline_supervised
from hltproject.score.score import compute_loss

logging.config.dictConfig(
    cutils.load_logger_config_file())
logger = logging.getLogger ( __name__ )

def output_directory ( dirname ):
    '''
     validates a name to be used as output directory
     
     \param dirname name of a directory to be used as output directory

     if dirname does not exists, creates it.
     if dirname is a pre-exitent directory, does nothing.
     Otherwise, raises an error
    '''
    if os.path.exists (dirname) and not os.path.isdir (dirname):
        raise argparse.ArgumentTypeError ("{} already exists and it is not a directory".format(dirname))
    os.makedirs (dirname, exist_ok=True)
    return dirname
    
def existing_file ( filename ):
    '''
     validates a name to be used as input file
     
     \param filename name of a file to be used as input file

     if filename does not exist, raises an error
     if filename exists and it is not a file, raises an error
    '''
    if not os.path.isfile (filename):
        raise argparse.ArgumentTypeError ("{} does not exist or it is not a file".format(filename))
    return filename

def _supervised ( args ):
    augment = args.augment
    train_fname = args.train
    validation_fname = args.validation
    test_fname = args.test
    output_fname = args.target + '/' + os.path.basename ( args.test ) + '.supervised_predictions'
    logger.info ("Computing predictions using supervised model.")
    logger.info (" input files: train {}, validation {} test {}".format(train_fname, validation_fname, test_fname) )
    augment_message = " input will " + ("" if args.augment else "not ") + "be augmented with pairwise dot products"
    logger.info ( augment_message )
    logger.info (" output file: {}".format(output_fname))
    baseline_supervised.compute_predictions ( train_fname, validation_fname, test_fname, augment, output_fname )
    

def _cosine ( args ):
    input_fname = args.test
    output_fname = args.target + '/' + os.path.basename ( args.test ) + '.cosine_predictions'
    logger.info ("Computing prediction using cosine. input file: {}, output file: {}".format(input_fname, output_fname))
    baseline_cosine.compute_predictions ( input_fname, output_fname )

def _compute_embeddings ( args ):
    input_fname = args.input
    output_fname = args.target + '/' + os.path.basename ( args.input )
    logger.info ("Computing ELMo embeddings. input file: {}, output base file: {}".format(input_fname, output_fname))
    compute_embeddings ( input_fname, output_fname )

def _compute_bert_embeddings ( args ):
    input_fname = args.input
    output_fname = args.target + '/' + os.path.basename ( args.input )
    logger.info ("Computing BERT embeddings. input file: {}, output base file: {}".format(input_fname, output_fname))
    compute_bert_embeddings ( input_fname, output_fname )

def _loss ( args ):
    model_fname = args.model
    gold_fname = args.input
    logger.info ("Computing loss for predictions: {}, original input file: {}".format(model_fname, gold_fname))
    compute_loss ( model_fname, gold_fname )
    

def main():
    parser = argparse.ArgumentParser(prog='hltproject')
    subparsers = parser.add_subparsers()



                              
    parser_compute_embeddings = subparsers.add_parser(
        'compute-embeddings', formatter_class=argparse.RawTextHelpFormatter,
        help='compute elmo embeddings of a given input dataset')
    parser_compute_embeddings.add_argument ('input', help='input filename', type=existing_file)
    parser_compute_embeddings.add_argument('-t', '--target', default='embeddings',
                        type=output_directory,        
                        help='target directory (default: embeddings)')
    parser_compute_embeddings.set_defaults(func=_compute_embeddings)


    parser_compute_bert_embeddings = subparsers.add_parser(
        'compute-bert-embeddings', formatter_class=argparse.RawTextHelpFormatter,
        help='compute bert embeddings of a given input dataset')
    parser_compute_bert_embeddings.add_argument ('input', help='input filename', type=existing_file)
    parser_compute_bert_embeddings.add_argument('-t', '--target', default='bert-embeddings',
                        type=output_directory,        
                        help='target directory (default: bert-embeddings)')
    parser_compute_bert_embeddings.set_defaults(func=_compute_bert_embeddings)







    
    parser_cosine = subparsers.add_parser(
        'baseline-cosine', formatter_class=argparse.RawTextHelpFormatter,
        help='compute prediction using on a cosine-based model')
    parser_cosine.add_argument ('test', help='test filename', type=existing_file)
    parser_cosine.add_argument('-t', '--target', default='predictions',
                            type=output_directory,
                            help='target directory (default: predictions)')
    parser_cosine.set_defaults(func=_cosine)
    
    parser_supervised = subparsers.add_parser(
        'baseline-supervised', formatter_class=argparse.RawTextHelpFormatter,
        help='compute prediction using on a supervised model')
    parser_supervised.add_argument ('train', help='train filename', type=existing_file)
    parser_supervised.add_argument ('validation', help='validation filename', type=existing_file)
    parser_supervised.add_argument ('test', help='test filename', type=existing_file)
    parser_supervised.add_argument('-g', '--augment', default=False,
                            action="store_true",
                            help='whether to augment input feature adding pairwise dot product or not')
    parser_supervised.add_argument('-t', '--target', default='predictions',
                            type=output_directory,
                            help='target directory (default: predictions)')
    parser_supervised.set_defaults(func=_supervised)
    
    parser_loss = subparsers.add_parser(
        'loss', formatter_class=argparse.RawTextHelpFormatter,
        help='compute loss for one prediction')
    parser_loss.add_argument ('model', help='model predictions', type=existing_file)
    parser_loss.add_argument ('input', help='input dataset', type=existing_file)
    parser_loss.set_defaults(func=_loss)
    
    args = parser.parse_args()
    args.func(args)
    
        
