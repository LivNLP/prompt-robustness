import sys
path = "./autoprompt"
sys.path.append(path)

import os
if os.path.exists(path):
    import autoprompt
    from autoprompt import create_trigger
else:
    assert False
    
import transformers
    
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import os

logger = logging.getLogger(__name__)


""" Dataset

    train: N sampled instances from original training data
        CB/MNLI: train_dp-{N}.tsv
    dev: 50 sampled instances for development
        CB/MNLI: dev.tsv
    test: Original development dataset (Used in robustness evaluation experiments)
        CB: val.tsv
        MNLI: dev_matched.tsv"

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
    parser.add_argument('--n-label-tokens', type=int, default=None) 
    parser.add_argument('--label-token-file', type=str, default=None)
    parser.add_argument('--result-filename', type=str, default=None) 
    parser.add_argument('--gpu', type=int, default=-1) 

    # LAMA-specific
    parser.add_argument('--tokenize-labels', action='store_true',
                        help='If specified labels are split into word pieces.'
                             'Needed for LAMA probe experiments.')
    parser.add_argument('--filter', action='store_true',
                        help='If specified, filter out special tokens and gold objects.'
                             'Furthermore, tokens starting with capital '
                             'letters will not appear in triggers. Lazy '
                             'approach for removing proper nouns.')
    parser.add_argument('--print-lama', action='store_true',
                        help='Prints best trigger in LAMA format.')

    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field')

    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='roberta-large',
                        help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use-ctx', action='store_true',
                        help='Use context sentences for relation extraction only')
    parser.add_argument('--perturbed', action='store_true',
                        help='Perturbed sentence evaluation of relation extraction: replace each object in dataset with a random other object')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-cand', type=int, default=10) # |V_candidate|
    parser.add_argument('--sentence-size', type=int, default=50) 
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    
    path = "/".join(args.result_filename.split("/")[:-1])
    os.makedirs(path, exist_ok=True)
    
    autoprompt.utils.set_seed(args.seed)
    autoprompt.create_trigger.run_model(args)
    