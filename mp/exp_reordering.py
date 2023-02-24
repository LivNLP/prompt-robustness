import myutils
from myutils import set_seed, get_mydataset_from_tsv, compute_accuracy, print_table
import os
import pandas as pd
import itertools
import copy
from openprompt.plms import load_plm

import logging
from logging import getLogger 
logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    # Configs
    args = myutils.get_base_args()
    exp_name = "mp-reordering"
    args.summary_filename = args.summary_filename.replace(".tsv", "_reordering.tsv")
    os.makedirs("/".join(args.summary_filename.split("/")[:-1]), exist_ok=True)
    pretrained_mp_dir = args.mp_dir
    eval_filename = myutils.get_eval_filename(args, "token_reordering")
    assert os.path.exists(pretrained_mp_dir), pretrained_mp_dir
    assert os.path.exists(args.data_dir), args.data_dir

    # Reordering prompt tokens
    reordered_token_list = [t for t in itertools.permutations(args.prompt_tokens, len(args.prompt_tokens))]
    reordered_token_list = reordered_token_list[1:] # Excluding original order

    # Evaluation
    result = []
    for seed in range(4):
        logging.info(f'Seed {seed}')
        
        # Load 
        model_dir = f"{pretrained_mp_dir}/{args.task}/{args.model_name}/{seed}/dp-{args.datapoint}/plm"
        assert os.path.exists(model_dir), model_dir
        logging.info('Loading model, tokenizer, etc.')
        plm, tokenizer, model_config, WrapperClass = load_plm(args.model_type, model_dir)

        # Reordering
        for reordered_tokens in reordered_token_list:
            template_text = copy.deepcopy(args.template)
            for idx in range(len(args.prompt_tokens)):
                template_text = template_text.replace(f"T{idx}", reordered_tokens[idx])

            logging.info(f"Prompt template: {template_text}")
            acc = compute_accuracy(args, seed, template_text, plm, tokenizer, 
                              model_config, WrapperClass, eval_filename)
            result.append([exp_name, args.task, args.model_name, seed, 
                           args.datapoint, template_text, acc])    

    
    # Save results
    df = pd.DataFrame(result, columns=["exp_name", "task", "model", "seed", "dp", "template", "acc"])
    df.to_csv(args.summary_filename, sep="\t")
    df = pd.read_csv(args.summary_filename, sep="\t")

    acc = df["acc"].mean() * 100
    column_names = ["Method", "Avg. Acc"]
    scores_list = [["Reordering (random)", f"{acc:.2f}"]]
    print_table(column_names, scores_list)

