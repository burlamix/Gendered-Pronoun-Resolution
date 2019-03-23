"""Welcome to hltproject.

This is the entry point of the application.
"""

import argparse
import os

import logging
import logging.config
import hltproject.utils.config as cutils

from hltproject.dataset_utils.compute_embeddings import compute_embeddings
from hltproject.baseline import baseline_cosine

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

def main():
    parser = argparse.ArgumentParser(prog='hltproject')
    subparsers = parser.add_subparsers()
                              
    parser_compute_embeddings = subparsers.add_parser(
        'compute-embeddings', formatter_class=argparse.RawTextHelpFormatter,
        help='compute elmo embeddings of a given input dataset')
    parser_compute_embeddings.add_argument ('input', help='input filename')
    parser_compute_embeddings.add_argument('-t', '--target', default='embeddings',
                        type=output_directory,        
                        help='target directory (default: embeddings)')
    parser_compute_embeddings.set_defaults(func=_compute_embeddings)
    
    parser_cosine = subparsers.add_parser(
        'baseline-cosine', formatter_class=argparse.RawTextHelpFormatter,
        help='compute prediction using on a cosine-based model')
    parser_cosine.add_argument ('test', help='test filename')
    parser_cosine.add_argument('-t', '--target', default='predictions',
                            type=output_directory,
                            help='target directory (default: predictions)')
    parser_cosine.set_defaults(func=_cosine)
    
    args = parser.parse_args()
    args.func(args)
    
        
