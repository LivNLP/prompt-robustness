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


def init_template(args, n_del):
    n_prompt_tokens = len(args.initial_trigger) - n_del
    assert n_prompt_tokens > 0
    # Special tokens <T> will be replaced with autoprompt tokens
    trigger_masks = " ".join(["[T]" for _ in range(n_prompt_tokens)]) 
    args.template = args.template.replace("[T]", trigger_masks)
    return args

    
def get_args_for_deletion(args, n_deletion=10, n_del_tokens = [3, 5, 7]):     
    # Reodering prompt tokens
    args_list = []
    
    n_del = 0
    new_args = copy.deepcopy(args)
    new_args = init_template(new_args, n_del) # Init template
    new_args.n_del = n_del
    new_args.del_idx = f"n/a"
    args_list.append(new_args)
    
    # init autoprompt tokens (single deletion)
    for del_idx in range(len(args.initial_trigger)):
        n_del = 1
        new_args = copy.deepcopy(args)
        new_args = init_template(new_args, n_del) # Init template
        new_args.n_del = n_del
        new_args.del_idx = f"{del_idx}"
        new_args.initial_trigger = [token for i, token in enumerate(args.initial_trigger) 
                                    if i != del_idx]
        args_list.append(new_args)

    # init autoprompt tokens (multiple deletion)
    for _ in range(n_deletion):
        for n_del in n_del_tokens:
            new_args = copy.deepcopy(args)
            new_args = init_template(new_args, n_del) # Init template
            del_idx = random.sample(list(range(len(args.initial_trigger))), n_del)

            new_args.n_del = n_del
            new_args.del_idx = "-".join(map(str, del_idx))
            new_args.initial_trigger = [token for i, token in enumerate(args.initial_trigger) 
                                        if i not in del_idx]
            args_list.append(new_args)
        
    return args_list
    
    
if __name__ == '__main__':    
    # Configs
    base_args = myutils.get_base_args()
    exp_name = "ap-deletion"
    base_args.summary_filename = base_args.summary_filename.replace(".tsv", "_deletion.tsv")
    os.makedirs("/".join(base_args.summary_filename.split("/")[:-1]), exist_ok=True)
    base_args.dev = Path(myutils.get_eval_filename(base_args, "token_deletion"))
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

        # Deletion
        args_list = get_args_for_deletion(args)
        
        for i, _args in enumerate(args_list):
            logger.info("New experiment")
            logger.info(f"Template: {_args.template}")
            prompt_template = myutils.get_prompt(_args)
            logger.info(f"Prompt template: {prompt_template}")
            acc = evaluate(_args, config, plm, tokenizer, device)
            result.append([exp_name, _args.task, _args.model_name, _args.seed, 
                           prompt_template, acc, _args.del_idx, _args.n_del])    
            
    # Save results
    df = pd.DataFrame(result, columns=["exp_name", "task", "model", "seed", 
                                       "template", "acc", "del_idx", "n_del"])
    df.to_csv(args.summary_filename, sep="\t")
    df = pd.read_csv(args.summary_filename, sep="\t")

    scores_list = []
    column_names = ["Method", "Avg. Acc"]
    
    for del_idx in range(base_args.xtrig):
        acc = df[df["del_idx"] == f"{del_idx}"]["acc"].mean() * 100
        scores_list.append([f"Single token deletion (idx={del_idx})", f"{acc:.2f}"])

    for n_del in [1, 3, 5, 7]:
        acc = df[df["n_del"] == n_del]["acc"].mean() * 100
        scores_list.append([f"Multiple token deletion (n={n_del})", f"{acc:.2f}"])    

    myutils.print_table(column_names, scores_list)
