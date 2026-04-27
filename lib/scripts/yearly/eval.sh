#!/bin/bash

CONFIG_PATH="${1:-configs/yearly/december_somsc_example.yaml}"

python eval_yearly_somsc.py --config "${CONFIG_PATH}" --save-csv
