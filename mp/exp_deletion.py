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
    exp_name = "mp-deletion"
    args.summary_filename = args.summary_filename.replace(".tsv", "_deletion.tsv")
    os.makedirs("/".join(args.summary_filename.split("/")[:-1]), exist_ok=True)
    pretrained_mp_dir = args.mp_dir
    eval_filename = myutils.get_eval_filename(args, "token_deletion")
    assert os.path.exists(pretrained_mp_dir), pretrained_mp_dir
    assert os.path.exists(args.data_dir), args.data_dir

    # Delete prompt tokens
    # [[PromptToken], [DeletedIndex]]
    deleted_prompt_token_list = [
        [["", "|", ","], [0]], 
        [["?", "", ","], [1]],
        [["?", "|", ""], [2]],
        [["", "|", ""], [0, 2]],
        [["", "", ","], [0, 1]],
        [["", "", ""], [0, 1, 2]],
        [["?", "", ""], [1, 2]],
    ]

    # Evaluation
    result = []
    for seed in range(4):
        logging.info(f'Seed {seed}')
        # Load 
        model_dir = f"{pretrained_mp_dir}/{args.task}/{args.model_name}/{seed}/dp-{args.datapoint}/plm"
        assert os.path.exists(model_dir), model_dir
        logging.info('Loading model, tokenizer, etc.')
        plm, tokenizer, model_config, WrapperClass = load_plm(args.model_type, model_dir)

        for deleted_tokens, del_idx in deleted_prompt_token_list:
            template_text = copy.deepcopy(args.template)
            for idx in range(len(args.prompt_tokens)):
                template_text = template_text.replace(f"T{idx}", deleted_tokens[idx])

            logging.info(f"Prompt template: {template_text}")
            acc = compute_accuracy(args, seed, template_text, plm, tokenizer, 
                              model_config, WrapperClass, eval_filename)
            result.append([exp_name, args.task, args.model_name, seed, 
                           args.datapoint, template_text, acc, 
                           "-".join(map(str, del_idx)), len(del_idx)])    

    
    # Save results
    df = pd.DataFrame(result, columns=["exp_name", "task", "model", "seed", "dp", "template", "acc", "del_idx", "n_del"])
    df.to_csv(args.summary_filename, sep="\t")
    df = pd.read_csv(args.summary_filename, sep="\t")

    scores_list = []
    column_names = ["Method", "Avg. Acc"]
    
    acc = df[df["del_idx"] == "0"]["acc"].mean() * 100
    scores_list.append(["Single token deletion (idx=0)", f"{acc:.2f}"])
    acc = df[df["del_idx"] == "1"]["acc"].mean() * 100
    scores_list.append(["Single token deletion (idx=1)", f"{acc:.2f}"])
    acc = df[df["del_idx"] == "2"]["acc"].mean() * 100
    scores_list.append(["Single token deletion (idx=2)", f"{acc:.2f}"])
    
    acc = df[df["n_del"] == 1]["acc"].mean() * 100
    scores_list.append(["Multiple token deletion (n=1)", f"{acc:.2f}"])
    acc = df[df["n_del"] == 2]["acc"].mean() * 100
    scores_list.append(["Multiple token deletion (n=2)", f"{acc:.2f}"])
    acc = df[df["n_del"] == 3]["acc"].mean() * 100
    scores_list.append(["Multiple token deletion (n=3)", f"{acc:.2f}"])
    
    acc = df[df["del_idx"] == "0"]["acc"].mean() * 100
    scores_list.append(["Multiple token deletion (one from front)", f"{acc:.2f}"])
    acc = df[df["del_idx"] == "0-1-2"]["acc"].mean() * 100
    scores_list.append(["Multiple token deletion (three from front)", f"{acc:.2f}"])
    
    acc = df[df["del_idx"] == "2"]["acc"].mean() * 100
    scores_list.append(["Multiple token deletion (one from back)", f"{acc:.2f}"])
    acc = df[df["del_idx"] == "0-1-2"]["acc"].mean() * 100
    scores_list.append(["Multiple token deletion (three from back)", f"{acc:.2f}"])
    print_table(column_names, scores_list)

