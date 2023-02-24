import os
import json
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from distutils.version import LooseVersion

import torch

import transformers
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer

import logging
logger = logging.getLogger(__name__)

import sys
sys.path.append("autoprompt")
assert os.path.exists("autoprompt")
import autoprompt
from autoprompt import create_trigger, utils

    
    
def load_pretrained(model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.eval()
    if LooseVersion(transformers.__version__) >= LooseVersion('4.0.0'):
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, 
                                                  use_fast= False) # Obtain the same behavior as transformers v3.x in v4.x
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    autoprompt.utils.add_task_specific_tokens(tokenizer)
    logger.info(f'{model.__class__.__name__} is loaded.')
    return config, model, tokenizer
    
    
def make_example_result(model_inputs, pred_labels, real_labels, tokenizer):     
    input_text = []
    for x_inp in model_inputs["input_ids"]:
        input_text.append(tokenizer.decode(x_inp))

    return [[x_inp, pred_label, real_label] for x_inp, pred_label, real_label 
            in zip(input_text, pred_labels, real_labels)]
    
    
def get_label_tokens(args):
    with open(args.label_map) as f: 
        all_label_map = json.load(f)
    label_map = {}
    for label, label_tokens in all_label_map.items():
        label_map[label] = label_tokens[:args.n_label_tokens]
    return label_map
    
    
def evaluate(args, config, model, tokenizer, device):
    logger.info('Setup {}'.format(args.method))
    
    if args.method == "autoprompt":
        label_map = get_label_tokens(args)        
        predictor = autoprompt.get_autoprompt_predictor(args, model, tokenizer, device)
        evaluation_fn = autoprompt.AccuracyFn(tokenizer, label_map, device)
        dev_loader = autoprompt.get_dev_loader(args, config, tokenizer, label_map)
        logger.info(f"Label map: {label_map}") 
        
    else:
        raise NotImplementedError
        
    logger.info('Evaluating')
    numerator = 0
    denominator = 0
    dev_example_results = [] 
    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            predict_logits = predictor(model_inputs)
            
        if args.method == "autoprompt":
            pred_labels, real_labels = evaluation_fn.get_predicted_label(predict_logits, labels) 
            dev_example_results += make_example_result(model_inputs, 
                                                       pred_labels, 
                                                       real_labels, 
                                                       tokenizer)
            numerator += evaluation_fn(predict_logits, labels).sum().item()
            denominator += labels.size(0) 
        else:
            assert False
        
    best_dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Best Dev metric: {best_dev_metric}')
    
    results = {"best_dev_metric": best_dev_metric, 
              "dev_example_results": dev_example_results} 
        
    #logger.info('Saving results')
    #save_results(args, results)
    
    return best_dev_metric
    
    
    
    
