import sys
path = "autoprompt"
sys.path.append(path)

import os
if os.path.exists(path):
    import autoprompt
    from autoprompt import label_search
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, help='JSON object defining label map')
    parser.add_argument('--initial-trigger', type=str, default=None, help='Manual prompt')
    parser.add_argument('--result-filename', type=str, default='result_label-token.tsv')
    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--k', type=int, default=50, help='Number of label tokens to print')
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--iters', type=int, default=10,
                        help='Number of iterations to run label search')
    parser.add_argument('--model-name', type=str, default='roberta-large',
                        help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-ctx', action='store_true',
                        help='Use context sentences.')
    
    parser.add_argument('--gpu', type=int, default=-1)
    
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
        
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    autoprompt.utils.set_seed(args.seed)
    autoprompt.label_search.main(args)
