# !/bin/bash

#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <expt_num>"
    exit 1
fi

EXPT_NUM=$1

rm /users/6/mehta423/daycent/data/v2.0/scaler_Y.pkl

python train_experiment.py --config configs/experiment_${EXPT_NUM}.yaml