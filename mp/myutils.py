import argparse
from pathlib import Path
import json
from prettytable import PrettyTable

import os
import openprompt # pip install openprompt==1.0.1
import datasets
from datasets import load_dataset
import pandas as pd
import numpy as np
import random
import torch
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification

import logging


def print_table(column_names, scores_list):
    tb = PrettyTable()
    tb.field_names = column_names
    for scores in scores_list:
        tb.add_row(scores)
    print(tb)


def get_mydataset_from_tsv(data_files):
    features = datasets.Features({
        'premise': datasets.Value('string'),
        'hypothesis': datasets.Value('string'),
        'idx': datasets.Value('int32'),
        'label': datasets.ClassLabel(names=['entailment', 'contradiction', 'neutral']),
    })

    #df = pd.read_csv(data_files["train"], sep="\t")
    #df = df.rename(columns={'sentence_A': 'premise', 
    #                        'sentence_B': 'hypothesis', 
    #                        'index': 'idx'})
    #raw_dataset_train = datasets.Dataset.from_pandas(
    #    df[['premise', 'hypothesis', 'label', 'idx']], features=features)

    df = pd.read_csv(data_files["test"], sep="\t")
    df = df.rename(columns={'sentence_A': 'premise', 
                            'sentence_B': 'hypothesis', 
                            'index': 'idx'})
    raw_dataset_val = datasets.Dataset.from_pandas(
        df[['premise', 'hypothesis', 'label', 'idx']], features=features)
    
    raw_dataset = {"test": raw_dataset_val}
    
    return raw_dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    
def compute_accuracy(args, seed, template_text, plm, tokenizer,
                     model_config, WrapperClass, eval_filename):
    data_files = {
        "test": eval_filename
    }
    raw_dataset = get_mydataset_from_tsv(data_files)

    dataset = {}
    dataset["test"] = []
    for data in raw_dataset["test"]:
        input_example = InputExample(text_a = data['premise'], 
                                     text_b = data['hypothesis'], 
                                     label=int(data['label']), 
                                     guid=data['idx'])
        dataset["test"].append(input_example)


    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
    #wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
    wrapped_tokenizer = WrapperClass(max_seq_length=args.max_seq_length, 
                                     decoder_max_length=3, 
                                     tokenizer=tokenizer,
                                     truncate_method="head")    
    model_inputs = {}
    model_inputs["test"] = []
    for sample in dataset["test"]:
        tokenized_example = wrapped_tokenizer.tokenize_one_example(
            mytemplate.wrap_one_example(sample), teacher_forcing=False
        )
        model_inputs["test"].append(tokenized_example)


    eval_dataloader = PromptDataLoader(dataset=dataset["test"], 
                                       template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, 
                                       max_seq_length=args.max_seq_length, decoder_max_length=3,
                                       batch_size=args.bsz, shuffle=False, teacher_forcing=False,
                                       predict_eos_token=False, truncate_method="head")

    # for example the verbalizer contains multiple label words in each class
    num_classes = len(args.label_words)
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes,
                                    label_words=args.label_words)

    prompt_model = PromptForClassification(plm=plm,template=mytemplate, 
                                           verbalizer=myverbalizer, freeze_plm=False)
    if args.gpu:
        prompt_model =  prompt_model.cuda()

    logging.info('Evaluating')    
    def myevaluate(prompt_model, eval_dataloader):
        allpreds = []
        alllabels = []
        for step, inputs in enumerate(eval_dataloader):
            if args.gpu:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return acc

    return myevaluate(prompt_model, eval_dataloader)


def get_eval_filename(args, eval_type):
    if args.task == "cb":
        if eval_type in ["token_reordering", "token_deletion"]:
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
        if eval_type in ["token_reordering", "token_deletion"]:
            return f"{args.data_dir}/superglue/{args.task}/dev_matched.tsv"
    
        elif eval_type == "adversarial_perturbation_label_no_change":
            return f"{args.data_dir}/superglue/{args.task}/perturbation-label-no-change.tsv"
        
        elif eval_type == "adversarial_perturbation_label_change":
            return f"{args.data_dir}/superglue/{args.task}/perturbation-label-change.tsv"
        
        elif eval_type == "no_adversarial_perturbation":
            return f"{args.data_dir}/superglue/{args.task}/no-perturbation.tsv"
        
        elif eval_type == "cross_dataset":
            return f"{args.data_dir}/superglue/cb/val.tsv"


def get_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['cb', 'mnli']) 
    parser.add_argument('--result-filename', type=str, default=None) 
    parser.add_argument('--bsz', type=int, default=16, help='Batch size')
    parser.add_argument('--mp-dir', type=str, default="model") 
    parser.add_argument('--data-dir', type=str, default="../data")
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    
    # MP Config
    if args.task == "cb":
        args.method = "manualprompt"
        args.model_type = "roberta"
        args.model_name = "roberta-large"
        args.datapoint = 200
        args.max_seq_length = 256
        args.label_words = [["yes"],    # label: 0
                            ["no"],     # label: 1
                            ["maybe"]]  # label: 2
        args.prompt_tokens = ["?", "|", ","]
        args.template = '{"placeholder":"text_b"}T0 T1 {"mask"}T2 {"placeholder":"text_a"}'
        
    elif args.task == "mnli":
        args.method = "manualprompt"
        args.model_type = "roberta"
        args.model_name = "roberta-large"
        args.datapoint = 200
        args.max_seq_length = 256
        args.label_words = [["yes"],    # label: 0
                            ["no"],     # label: 1
                            ["maybe"]]  # label: 2
        args.prompt_tokens = ["?", "|", ","]
        args.template = '{"placeholder":"text_b"}T0 T1 {"mask"}T2 {"placeholder":"text_a"}'
        
    args.summary_filename = f"analysis-result-{args.task}/{args.model_name}/summary-result.tsv"
    return args



