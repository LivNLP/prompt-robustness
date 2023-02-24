# Evaluating the Robustness of Discrete Prompts
Yoichi Ishibashi, Danushka Bollegala, Katsuhito Sudoh, Satoshi Nakamura: [Evaluating the Robustness of Discrete Prompts](https://arxiv.org/abs/2302.05619) (EACL 2023)

## Usage
Our experiment is divided into two phases (1) prompt learning (2) analyzing the robustness of the learned prompts.


1. Learning prompt tokens by AutoPrompt (AP).
```bash
cd ap
sh ap_label-token-search.sh
sh ap_trigger-token-search.sh
```

2. Fine-tuning PLM by Manually-written Prompts (MP).
```bash
cd mp
sh mp_finetuning.sh
```

3. Evaluating the robustness of LM prompt
The following scripts perform the four robustness evaluations of LM prompts.

AutoPrompt (AP)
```bash
cd ap
sh ap_run-all-robust-eval.sh 
```

Manually-written Prompts (MP) 
```bash
cd mp
sh mp_run-all-robust-eval.sh 
```

## The adversarial NLI dataset
We created the adversarial NLI dataset (see 3.5 Adversarial Perturbations in our paper). These datasets were used for the prompt robustness experiments described above.
```bash
data/superglue/cb/perturbation-label-change.tsv
data/superglue/cb/perturbation-label-no-change.tsv
data/superglue/mnli/perturbation-label-change.tsv
data/superglue/mnli/perturbation-label-no-change.tsv
```

## External Libraries
- [OpenPrompt](https://github.com/thunlp/OpenPrompt)
- [autoprompt](https://github.com/ucinlp/autoprompt)


## Citation
```bibtex
@inproceedings{Ishibashi:EACL:2023,
  author = {Yoichi Ishibashi and Danushka Bollegala and Katsuhito Sudoh and Satoshi Nakamura},  
  title = {Evaluating the Robustness of Discrete Prompts},
  booktitle = {Proc. of  the 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2023)},
  year = {2023}
}
```