import os
import random
import copy
import torch
import logging
from evaluate_prompt import evaluate, load_pretrained
import myutils
import pandas as pd
from pathlib import Path
    
import logging
from logging import getLogger 
logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def init_template(args):
    n_prompt_tokens = len(args.initial_trigger)
    # Special tokens <T> will be replaced with autoprompt tokens
    trigger_masks = " ".join(["[T]" for _ in range(n_prompt_tokens)]) 
    args.template = args.template.replace("[T]", trigger_masks)
    return args

    
def get_args(args): 
    # Init template
    args = init_template(args)    
    return args
    
    
if __name__ == '__main__':    
    # Configs
    base_args = myutils.get_base_args()
    exp_name = "ap-original-acc"
    base_args.summary_filename = base_args.summary_filename.replace(".tsv", "_original-acc.tsv")
    os.makedirs("/".join(base_args.summary_filename.split("/")[:-1]), exist_ok=True)
    base_args.dev = Path(myutils.get_eval_filename(base_args, "original-acc"))
    assert os.path.exists(base_args.data_dir), base_args.data_dir
    
    # Evaluation
    result = []
    for seed in range(4):
        logging.info(f'Seed {seed}')
        
        # Load PLM
        logger.info('Loading model, tokenizer, etc.')
        myutils.set_seed(seed)
        device = torch.device('cuda:{}'.format(base_args.gpu) if torch.cuda.is_available() else 'cpu')
        config, plm, tokenizer = load_pretrained(base_args.model_name)
        plm.to(device)
        
        # Load autoprompt tokens and the label map
        logger.info('Loading autoprompt.')
        args = copy.deepcopy(base_args)
        args = myutils.load_autoprompt(args, seed)

        # Reordering
        _args = get_args(args)
        
        logger.info("New experiment")
        logger.info(f"Template: {_args.template}")
        prompt_template = myutils.get_prompt(_args)
        logger.info(f"Prompt template: {prompt_template}")
        acc = evaluate(_args, config, plm, tokenizer, device)
        result.append([exp_name, _args.task, _args.model_name, 
                       _args.seed, prompt_template, acc]) 
    
    # Save results
    df = pd.DataFrame(result, columns=["exp_name", "task", "model", "seed", "template", "acc"])
    df.to_csv(args.summary_filename, sep="\t")
    df = pd.read_csv(args.summary_filename, sep="\t")
    
    acc = df["acc"].mean() * 100
    column_names = ["Method", "Avg. Acc"]
    scores_list = [["Orignal", f"{acc:.2f}"]]
    myutils.print_table(column_names, scores_list)

    
    
    
    
    