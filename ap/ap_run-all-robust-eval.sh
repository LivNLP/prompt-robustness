#!/bin/bash

task="cb"
python exp_original-accuracy.py --task "$task"
python exp_reordering.py --task "$task"
python exp_deletion.py --task "$task"
python exp_cross-dataset.py --task "$task"
python exp_adv-perturbation.py --task "$task"


task="mnli"
python exp_original-accuracy.py --task "$task"
python exp_reordering.py --task "$task"
python exp_deletion.py --task "$task"
python exp_cross-dataset.py --task "$task"
python exp_adv-perturbation.py --task "$task"


