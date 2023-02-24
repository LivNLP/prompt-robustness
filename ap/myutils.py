import argparse
from prettytable import PrettyTable
import random
import numpy as np
import torch
import json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def print_table(column_names, scores_list):
    tb = PrettyTable()
    tb.field_names = column_names
    for scores in scores_list:
        tb.add_row(scores)
    print(tb)
    
    
def get_prompt(args):
    prompt = args.template
    for token in args.initial_trigger:
        prompt = prompt.replace("[T]", token, 1)
    return prompt


def load_autoprompt(args, seed):
    # Set seed
    args.seed = seed
    seed_dir = f"{args.ap_dir}/{args.task}/{args.model_name}/{args.seed}"
    
    # Load pre-trained label tokens
    args.label_map = f"{seed_dir}/result_label-tokens_xtrig-{args.xtrig}/label-tokens_datapoint-{args.datapoint}.json"
    
    # Load pre-trained trigger tokens
    trigger_token_file = f"{seed_dir}/result_trigger-tokens_xtrig-{args.xtrig}_vy-{args.n_label_tokens}_vcand-{args.vcand}/trigger-tokens_datapoint-{args.datapoint}.json"
    with open(trigger_token_file) as f:
        initial_trigger = json.load(f)["best_trigger_tokens"]
    args.initial_trigger = initial_trigger
    return args
    
    
def get_eval_filename(args, eval_type):
    if args.task == "cb":
        if eval_type in ["token_reordering", "token_deletion", "original-acc"]:
            return f"{args.data_dir}/superglue/{args.task}/val.tsv"
    
        elif eval_type == "adversarial_perturbation_label_no_change":
            return f"{args.data_dir}/superglue/{args.task}/perturbation-label-no-change.tsv"
        
        elif eval_type == "adversarial_perturbation_label_change":
            return f"{args.data_dir}/superglue/{args.task}/perturbation-label-change.tsv"

        elif eval_type == "no_adversarial_perturbation":
            return f"{args.data_dir}/superglue/{args.task}/no-perturbation.tsv"
    
        elif eval_type == "cross_dataset":
            return f"{args.data_dir}/superglue/mnli/dev_matched.tsv"
        
    elif args.task == "mnli":
        if eval_type in ["token_reordering", "token_deletion", "original-acc"]:
            return f"{args.data_dir}/superglue/{args.task}/dev_matched.tsv"
    
        elif eval_type == "adversarial_perturbation_label_no_change":
            return f"{args.data_dir}/superglue/{args.task}/perturbation-label-no-change.tsv"
        
        elif eval_type == "adversarial_perturbation_label_change":
            return f"{args.data_dir}/superglue/{args.task}/perturbation-label-change.tsv"
        
        elif eval_type == "no_adversarial_perturbation":
            return f"{args.data_dir}/superglue/{args.task}/no-perturbation.tsv"
        
        elif eval_type == "cross_dataset":
            return f"{args.data_dir}/superglue/cb/val.tsv"
    assert False, f"eval_type:{eval_type}, args.task:{args.task}"



def get_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['cb', 'mnli']) 
    #parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
    #parser.add_argument('--n-label-tokens', type=int, default=None) 
    parser.add_argument('--result-filename', type=str, default=None) 
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--ap-dir', type=str, default="model") 
    parser.add_argument('--data-dir', type=str, default="../data")
    parser.add_argument('--gpu', type=int, default=0)  

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

    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    #parser.add_argument('--model-name', type=str, default='bert-base-cased',
    #                    help='Model name passed to HuggingFace AutoX classes.')
    #parser.add_argument('--seed', type=int, default=model_seed)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use-ctx', action='store_true',
                        help='Use context sentences for relation extraction only')
    parser.add_argument('--perturbed', action='store_true',
                        help='Perturbed sentence evaluation of relation extraction: replace each object in dataset with a random other object')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-cand', type=int, default=10)
    parser.add_argument('--sentence-size', type=int, default=50)  
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    

    # AP settings
    if args.task == "cb":
        args.method = "autoprompt"
        args.model_name = "roberta-large"
        args.datapoint = 200
        args.xtrig = 10
        args.n_label_tokens = 3
        args.vcand = 10
        args.template = '<s> {sentence_A} [P] [T] {sentence_B} </s>' 
        # [P]: Token for label prediction 
        # This is [MASK] token, but denoted by [P] in the template to distinguish it from other [MASK] tokens that might appear.
        # [T]: Trigger token (AutoPrompt token)
        # Number of [T] will be changed by the type of perturbations
        
    elif args.task == "mnli":
        args.method = "autoprompt"
        args.model_name = "roberta-large"
        args.datapoint = 200
        args.xtrig = 10
        args.n_label_tokens = 3
        args.vcand = 10
        args.template = '<s> {sentence_A} [P] [T] {sentence_B} </s>' 
        
    args.summary_filename = f"analysis-result-{args.task}/{args.model_name}/summary-result.tsv"
    return args



