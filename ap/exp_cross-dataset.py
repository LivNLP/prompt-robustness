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



if __name__ == '__main__':    
    # Configs
    base_args = myutils.get_base_args()
    exp_name = "ap-cross-dataset"
    base_args.summary_filename = base_args.summary_filename.replace(".tsv", "_cross-dataset.tsv")
    os.makedirs("/".join(base_args.summary_filename.split("/")[:-1]), exist_ok=True)
    base_args.dev = Path(myutils.get_eval_filename(base_args, "cross_dataset"))
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

        # Cross-dataset eval.
        logger.info(f"Template: {args.template}")
        prompt_template = myutils.get_prompt(args)
        logger.info(f"Prompt template: {prompt_template}")
        acc = evaluate(args, config, plm, tokenizer, device)
        result.append([exp_name, args.task, args.model_name, args.seed, 
                       prompt_template, acc])    
            
    # Save results
    df = pd.DataFrame(result, columns=["exp_name", "task", "model", "seed", "template", "acc"])
    df.to_csv(args.summary_filename, sep="\t")
    df = pd.read_csv(args.summary_filename, sep="\t")

    acc = df["acc"].mean() * 100
    column_names = ["Eval Dataset", "Avg. Acc"]
    if args.task == "cb":
        scores_list = [["MNLI", f"{acc:.2f}"]]
    elif args.task == "mnli":
        scores_list = [["CB", f"{acc:.2f}"]]
    myutils.print_table(column_names, scores_list)




