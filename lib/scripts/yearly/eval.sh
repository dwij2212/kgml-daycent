#!/bin/bash

CONFIG_PATH="${1:-configs/yearly/experiment_state_v1.yaml}"

python eval_yearly_somsc.py --config "${CONFIG_PATH}" --save-csv
