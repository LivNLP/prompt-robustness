# ==============================================================================
# This source code is derived from AutoPrompt source code.
# https://github.com/ucinlp/autoprompt
# ==============================================================================

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import autoprompt.utils as utils
import transformers
from distutils.version import LooseVersion

logger = logging.getLogger(__name__)


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model, trigger_ids): 
        self._model = model
        self.trigger_ids = trigger_ids

    def __call__(self, model_inputs):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')
        model_inputs = replace_trigger_tokens(model_inputs, self.trigger_ids, trigger_mask)
        
        if LooseVersion(transformers.__version__) >= LooseVersion('4.0.0'):
            logits = self._model(**model_inputs).logits 
        else:
            logits, *_ = self._model(**model_inputs)
        
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, label_map, device, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        #logger.info(label_map)
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(utils.encode_label(tokenizer, label_tokens, tokenize_labels).to(device))
            self._pred_to_label.append(label)
        #logger.info(self._all_label_ids)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        
        # Add up the number of entries where loss is greater than or equal to gold_logp.
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # less than in case of num. prec. issues

        return correct.float() 

    # TODO: @rloganiv - This is hacky. Replace with something sensible.
    def predict(self, predict_logits):
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        return predictions
    
    def label_token_ids_to_label(self, label_ids):
        real_labels = []
        for l in label_ids:
            for label, label_token_ids in zip(self._pred_to_label, self._all_label_ids):
                assert label_token_ids.shape[0] == 1
                label_token_ids = label_token_ids[0]
                if torch.equal(label_token_ids, l):
                    real_labels.append(label)
        assert np.array(real_labels).shape[0] == label_ids.shape[0], "{} != {}".format(np.array(real_labels).shape, label_ids.shape)
        return real_labels
    
    def get_predicted_label(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()] # ['neutral', 'entailment', ...]

        real_label = self.label_token_ids_to_label(gold_label_ids) # ['neutral', 'entailment', ...]

        return predictions, real_label


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    return out


def get_loss(predict_logits, label_ids):
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp
        
            
def get_autoprompt_predictor(args, model, tokenizer, device):    
    # Obtain the initial trigger tokens and label mapping
    trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)
    logger.debug(f'Initial trigger: {args.initial_trigger}')
    logger.debug(f'Trigger ids: {trigger_ids}')    
        
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    predictor = PredictWrapper(model, trigger_ids) 

    return predictor


def get_dev_loader(args, config, tokenizer, label_map):
    templatizer = utils.TriggerTemplatizer( 
        args.template,
        config,
        tokenizer,
        label_map=label_map,
        label_field=args.label_field,
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=False,
        use_ctx=args.use_ctx
    )
    
    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    dev_dataset = utils.load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)        
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)
    return dev_loader
    

    

            
            
