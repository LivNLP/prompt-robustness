#!/bin/bash

task="cb"
python exp_reordering.py --mp-dir ./model --task "$task" --gpu
python exp_deletion.py --mp-dir ./model --task "$task" --gpu
python exp_adv-perturbation.py --mp-dir ./model --task "$task" --gpu
python exp_cross-dataset.py --mp-dir ./model --task "$task" --gpu

task="mnli"
python exp_reordering.py --mp-dir ./model --task "$task" --gpu
python exp_deletion.py --mp-dir ./model --task "$task" --gpu
python exp_adv-perturbation.py --mp-dir ./model --task "$task" --gpu
python exp_cross-dataset.py --mp-dir ./model --task "$task" --gpu
