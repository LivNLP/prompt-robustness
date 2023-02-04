# Usage
Our experiment is divided into two phases (1) prompt learning (2) analyzing the robustness of the learned prompts.


1. Learning prompt tokens by AP.
```sh
sh ap_trigger-token-search.sh
```

2. Fine-tuning by MP.
```sh
sh mp_fine-tuning.sh
```


3. We can run four experiments on the robustness of AP/MP with the following scripts.
```sh
sh exp_run-all-robust-eval.sh 
```

# The adversarial NLI dataset
We created the adversarial NLI dataset (See 3.5 Adversarial Perturbations in our paper).
This dataset was used for the experiments on the robustness of prompts described above.
```
data/superglue/cb/val_perturbation-label-change.tsv
data/superglue/cb/val_perturbation-label-no-change.tsv

data/superglue/mnli/val_perturbation-label-change.tsv
data/superglue/mnli/val_perturbation-label-no-change.tsv
```

# External Libraries
The two packages below are open libraries and are not produced by us.
- OpenPrompt
- autoprompt



