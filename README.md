# DayCent Selection Experiments

This repository contains the DayCent emulator code used to study spatial
training-location selection, including incremental baselines and graph-based
Bayesian optimization over fixed-size location subsets.

Most of the code you need is in [`lib/`](lib/). Start with
[`lib/README.md`](lib/README.md), then read:

1. [`lib/configs/selection/selection_exp6.yaml`](lib/configs/selection/selection_exp6.yaml)
2. [`lib/run_bo_experiment.py`](lib/run_bo_experiment.py)
3. [`lib/selection/bo_graph.py`](lib/selection/bo_graph.py)
4. [`lib/run_ensemble_experiment.py`](lib/run_ensemble_experiment.py)
5. [`lib/selection/`](lib/selection/)

Set up and activate your own Python environment first. Install Torch plus the
minimum packages in [`requirements-selection.txt`](requirements-selection.txt),
update the data paths in the selection YAML, and run the Python entry points
from `lib/`.

For the incremental baselines, start with:

```bash
cd lib
bash scripts/selection/run_all.sh
```
