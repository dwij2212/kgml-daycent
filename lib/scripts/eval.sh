#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <expt_num>"
    exit 1
fi

EXPT_NUM=$1

rm -rf ../output/v2.0/plots/

python eval_experiment.py --config "configs/experiment_${EXPT_NUM}.yaml"