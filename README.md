# DayCent emulator V1

This repository contains the trained crop-yield and autoregressive soil organic
carbon (SOC) emulators, their training and evaluation code, and the recorded
site-split experiments. The package and its assets are in **`emulator_V1/`**.

## Where to start

| Task | Documentation or files |
|---|---|
| Understand the models and reproduce the experiments | [Package guide](emulator_V1/README.md) |
| Predict yield and SOC for new inputs | [Inference guide](emulator_V1/emulator_inference/README.md) |
| Review training defaults and data paths | [Configuration](emulator_V1/config.yml) |
| Review exact site splits, normalization and run settings | [Run records](emulator_V1/runs/README.md) |
| Compare recorded results | [Results summary](emulator_V1/results/summary.csv) |
| Inspect training and evaluation code | [Python package](emulator_V1/daycent_emulator/) |

Two trained checkpoints are included:

| Model | Site-split seed | Model/training seed | Checkpoint |
|---|---:|---:|---|
| Yield: corn, soybean and winter wheat | 20260915 | 20261009 | [yield.pt](emulator_V1/emulator_inference/weights/yield.pt) |
| SOC: annual autoregressive carbon-pool rollout | 20260914 | 20261008 | [soc.pt](emulator_V1/emulator_inference/weights/soc.pt) |

Configurations and results are retained for all three yield splits and both SOC
splits. Other splits can be retrained using their saved recipes. The selected
checkpoints were chosen from recorded test performance on different site splits;
the [package guide](emulator_V1/README.md) explains the selection and score definitions.

## Set up the environment

Use Python 3.12. Run these commands from the repository root:

```bash
python3.12 -m venv ../daycent-env
source ../daycent-env/bin/activate
python -m pip install -r emulator_V1/requirements.txt
python emulator_V1/run.py --help
```

The requirements pin the original training environment, including PyTorch
2.7.1+cu118 and NumPy 2.4.6. Inference supports CPU and CUDA; training requires
CUDA. The recorded training runs used an NVIDIA A100 40GB.

## Predict from new inputs

Supply daily weather, initial site conditions and management schedules in the
formats described in the [inference guide](emulator_V1/emulator_inference/README.md).
This workflow uses the included weights and normalization; PBM output labels and
the original training dataset are not required.

```bash
python emulator_V1/emulator_inference/predict.py \
  --target both \
  --weather-dir /path/to/new_inputs/Weather \
  --site-conditions /path/to/new_inputs/initial_site_conditions.xlsx \
  --management /path/to/new_inputs/management.csv \
  --first-year 2000 --last-year 2023 \
  --output /path/to/new_predictions \
  --device cpu
```

SOC starts from the supplied initial pools and carries predictions through each
subsequent year. Yield uses planting and harvest events plus prior driver history.
The inference guide documents the required history, units and output CSVs.

## Reproduce the saved experiments

The original Round-3 dataset is supplied separately. Build the prepared inputs
from that dataset, then evaluate the included checkpoints:

```bash
python emulator_V1/run.py prepare-data \
  --raw-root /path/to/KGML_Midwest_Daily_2026 \
  --output /path/to/prepared_data --workers 2

python emulator_V1/run.py evaluate --target yield --site-split-seed 20260915 \
  --input-root /path/to/prepared_data \
  --output /path/to/evaluations/yield_split20260915 --device cuda:0

python emulator_V1/run.py evaluate --target soc --site-split-seed 20260914 \
  --input-root /path/to/prepared_data \
  --output /path/to/evaluations/soc_split20260914 --device cuda:0
```

Use `--device cpu` for CPU evaluation. Data preparation verifies all 777 required
files against the included input manifest. Use new output directories outside
`emulator_V1/` to preserve the packaged weights and results.

For training from random initialization, use `run.py train` with the target,
retained site-split seed, prepared-input path and a new output directory. See the
[reproduction instructions](emulator_V1/README.md#reproduction-seeds-and-split)
for complete yield and SOC commands, training budgets and seed definitions.

To view the recorded results without running evaluation:

```bash
python emulator_V1/run.py results
```
