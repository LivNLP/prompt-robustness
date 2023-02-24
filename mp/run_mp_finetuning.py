# ==============================================================================
# This source code is derived from OpenPrompt source code.
# https://github.com/thunlp/OpenPrompt
# ==============================================================================


import os    
import datasets
from datasets import load_dataset
import pandas as pd
import argparse
import json
import numpy as np
import random
import torch

import openprompt
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from transformers import  AdamW, get_linear_schedule_with_warmup
from openprompt.prompts import ManualVerbalizer

import logging
from logging import getLogger 
logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_mydataset_from_tsv(data_files):
    features = datasets.Features({
        'premise': datasets.Value('string'),
        'hypothesis': datasets.Value('string'),
        'idx': datasets.Value('int32'),
        'label': datasets.ClassLabel(names=['entailment', 'contradiction', 'neutral']),
    })

    df = pd.read_csv(data_files["train"], sep="\t")
    df = df.rename(columns={'sentence_A': 'premise', 
                            'sentence_B': 'hypothesis', 
                            'index': 'idx'})
    raw_dataset_train = datasets.Dataset.from_pandas(
        df[['premise', 'hypothesis', 'label', 'idx']], features=features)

    
    df = pd.read_csv(data_files["validation"], sep="\t")
    df = df.rename(columns={'sentence_A': 'premise', 
                            'sentence_B': 'hypothesis', 
                            'index': 'idx'})
    raw_dataset_val = datasets.Dataset.from_pandas(
        df[['premise', 'hypothesis', 'label', 'idx']], features=features)
    
    df = pd.read_csv(data_files["test"], sep="\t")
    df = df.rename(columns={'sentence_A': 'premise', 
                            'sentence_B': 'hypothesis', 
                            'index': 'idx'})
    raw_dataset_test = datasets.Dataset.from_pandas(
        df[['premise', 'hypothesis', 'label', 'idx']], features=features)
    
    raw_dataset = {"train": raw_dataset_train,
                    "validation": raw_dataset_val,
                   "test": raw_dataset_test}
    
    return raw_dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)        


def main(args):
    task = args.task
    seed = args.seed
    dp = args.dp
    model_type = args.model_type
    model_name = args.model_name
    weight_decay_config = args.weight_decay_config
    lr = args.lr
    iters = args.iters
    bsz = args.bsz 
    max_seq_length = args.max_seq_length
    template_text = args.template_text
    eval_interval = args.eval_interval
    label_words = args.label_words
    num_classes = args.num_classes
    out_path = args.out_path


    """ Dataset

        train: N sampled instances from original training data
            CB/MNLI: train_dp-{N}.tsv
        dev: 50 sampled instances for development
            CB/MNLI: dev.tsv
        test: Original development dataset (Used in robustness evaluation experiments)
            CB: val.tsv
            MNLI: dev_matched.tsv"

    """
    if task == "mnli":
        data_files = {"train": f"../data/superglue/{task}/datapoint-random-{seed}/train_dp-{dp}.tsv",
                      "validation": f"../data/superglue/{task}/val.tsv",
                      "test": f"../data/superglue/{task}/dev_matched.tsv"}
        raw_dataset = get_mydataset_from_tsv(data_files)
    elif task == "cb":
        data_files = {"train": f"../data/superglue/{task}/datapoint-random-{seed}/train_dp-{dp}.tsv",
                      "validation": f"../data/superglue/{task}/dev.tsv",
                      "test": f"../data/superglue/{task}/val.tsv"}
        raw_dataset = get_mydataset_from_tsv(data_files)

    # In this scripts, you will learn
    # 1. how to use integrate huggingface datasets utilities into openprompt to
    #  enable prompt learning in diverse datasets.
    # 2. How to instantiate a template using a template language
    # 3. How does the template wrap the input example into a templated one.
    # 4. How do we hide the PLM tokenization details behind and provide a simple tokenization
    # 5. How do construct a verbalizer using one/many label words
    # 5. How to train the prompt like a traditional Pretrained Model.

    # Note that if you are running this scripts inside a GPU cluster, there are chances are you are not able to connect to huggingface website directly.
    # In this case, we recommend you to run `raw_dataset = load_dataset(...)` on some machine that have internet connections.
    # Then use `raw_dataset.save_to_disk(path)` method to save to local path.
    # Thirdly upload the saved content into the machiine in cluster.
    # Then use `load_from_disk` method to load the dataset.


    dataset = {}
    for split in ['train', "validation", "test"]:
        dataset[split] = []
        for data in raw_dataset[split]:
            input_example = InputExample(text_a = data['premise'], 
                                         text_b = data['hypothesis'], 
                                         label=int(data['label']), 
                                         guid=data['idx'])
            dataset[split].append(input_example)

    # You can load the plm related things provided by openprompt simply by calling:
    plm, tokenizer, model_config, WrapperClass = load_plm(model_type, model_name)

    # Constructing Template
    # A template can be constructed from the yaml config, but it can also be constructed by directly passing arguments.
    #template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

    # To better understand how does the template wrap the example, we visualize one instance.
    #wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])

    # Now, the wrapped example is ready to be pass into the tokenizer, hence producing the input for language models.
    # You can use the tokenizer to tokenize the input by yourself, but we recommend using our wrapped tokenizer, which is a wrapped tokenizer tailed for InputExample.
    # The wrapper has been given if you use our `load_plm` function, otherwise, you should choose the suitable wrapper based on
    # the configuration in `openprompt.plms.__init__.py`.
    # Note that when t5 is used for classification, we only need to pass <pad> <extra_id_0> <eos> to decoder.
    # The loss is calcaluted at <extra_id_0>. Thus passing decoder_max_length=3 saves the space
    wrapped_tokenizer = WrapperClass(max_seq_length=max_seq_length, 
                                     decoder_max_length=3, 
                                     tokenizer=tokenizer,
                                     truncate_method="head")
    # or
    #from openprompt.plms import T5TokenizerWrapper
    #wrapped_tokenizer= T5TokenizerWrapper(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")

    # You can see what a tokenized example looks like by
    #tokenized_example = wrapped_tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)

    # Now it's time to convert the whole dataset into the input format!
    # Simply loop over the dataset to achieve it!
    #model_inputs = {}
    #for split in ['train', 'validation', 'test']:
    #    model_inputs[split] = []
    #    for sample in dataset[split]:
    #        tokenized_example = wrapped_tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
    #        model_inputs[split].append(tokenized_example)


    # We provide a `PromptDataLoader` class to help you do all the above matters and wrap them into an `torch.DataLoader` style iterator.

    train_dataloader = PromptDataLoader(dataset=dataset["train"], 
                                        template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, 
                                        max_seq_length=max_seq_length, decoder_max_length=3,
                                        batch_size=bsz, shuffle=True, teacher_forcing=False, 
                                        predict_eos_token=False, truncate_method="head")

    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], 
                                             template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, 
                                             max_seq_length=max_seq_length, decoder_max_length=3,
                                             batch_size=bsz, shuffle=False, teacher_forcing=False, 
                                             predict_eos_token=False, truncate_method="head")
    
    test_dataloader = PromptDataLoader(dataset=dataset["test"], 
                                             template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, 
                                             max_seq_length=max_seq_length, decoder_max_length=3,
                                             batch_size=bsz, shuffle=False, teacher_forcing=False, 
                                             predict_eos_token=False, truncate_method="head")

    # Define the verbalizer
    # In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:

    # for example the verbalizer contains multiple label words in each class
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes,
                                    label_words=label_words)
    # Although you can manually combine the plm, template, verbalizer together, we provide a pipeline
    # model which take the batched data from the PromptDataLoader and produce a class-wise logits

    prompt_model = PromptForClassification(plm=plm,template=mytemplate, 
                                           verbalizer=myverbalizer, freeze_plm=False)
    if args.gpu >= 0:
        prompt_model =  prompt_model.cuda()

    # Now the training is standard
    loss_func = torch.nn.CrossEntropyLoss()


    if weight_decay_config == "config1":
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    elif weight_decay_config == "config2":
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=lr, 
                          eps=adam_epsilon, 
                          weight_decay=weight_decay
                         )

    def myevaluate(prompt_model, validation_dataloader):
        allpreds = []
        alllabels = []
        for step, inputs in enumerate(validation_dataloader):
            if args.gpu >= 0:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return acc

    logging.info("Training")
    best_acc = -1
    i = 0
    iter_per_epoch = raw_dataset["train"].num_rows / bsz
    n_itr = int(iters / iter_per_epoch)
    assert n_itr > 0
    for itr in range(n_itr):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            i+=1
            if args.gpu >= 0:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if i % eval_interval == 0:
                dev_acc = myevaluate(prompt_model, validation_dataloader)
                epoch = int(i / iter_per_epoch)
                logging.info(f"Epoch {epoch}, Iter {i}, Average loss: {tot_loss/(step+1)}, Dev acc: {dev_acc}")

    acc = myevaluate(prompt_model, validation_dataloader)
    prompt_model.prompt_model.plm.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    logging.info('Done')

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dp', required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--eval_interval', type=int, required=True)
    parser.add_argument('--weight_decay_config', type=str, default="config1")
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--bsz', type=int, default=4, help='Batch size')
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--template_text', type=str, default='{"placeholder":"text_b"}? | {"mask"}, {"placeholder":"text_a"}')
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()
        
    os.makedirs(args.out_path, exist_ok=True)
    
    args.label_words = [["yes"],    # label: 0
                        ["no"],     # label: 1
                        ["maybe"]]  # label: 2
    args.num_classes = len(args.label_words)
    with open("{}/args.json".format(args.out_path), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    set_seed(args.seed)
    main(args)    
