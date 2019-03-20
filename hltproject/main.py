"""Welcome to hltproject.

This is the entry point of the application.
"""

import argparse
import os

from hltproject.dataset_utils.compute_embeddings import compute_embeddings

def _compute_embeddings ( args ):
    input_fname = args.input
    output_fname = args.target + '/' + os.path.basename ( args.input )
    compute_embeddings ( input_fname, output_fname )

def main():
    parser = argparse.ArgumentParser(prog='hltproject')
    subparsers = parser.add_subparsers()
                              
    parser_compute_embeddings = subparsers.add_parser(
        'compute-embeddings', formatter_class=argparse.RawTextHelpFormatter,
        help='compute elmo embeddings of a given input dataset')
    parser_compute_embeddings.add_argument ('input', help='input filename')
    parser_compute_embeddings.add_argument('-t', '--target', default='embeddings',
                              help='target directory (default embeddings)')
    parser_compute_embeddings.set_defaults(func=_compute_embeddings)
    
    args = parser.parse_args()
    args.func(args)
    
        
