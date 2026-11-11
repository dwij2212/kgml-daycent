# DayCent emulator — V1 (v1.0.0)

One package for the final **yield and SOC** models and their random-location split repeats. It contains two selected checkpoints, model and data-loading code, training/evaluation recipes, exact split membership, normalization statistics, and recorded results. The historical reference split for both outputs is **20260914**; the selected inference checkpoints use **20260915 for yield** and **20260914 for SOC**.

Configurations and results for all three yield runs and two SOC runs are retained in `runs/`. Duplicate checkpoints and runtime caches have been removed. Temporal experiments and alternative model experiments are outside this release.

For prediction on **new inputs**, use [emulator_inference/README.md](emulator_inference/README.md). That standalone subfolder contains the selected yield and SOC checkpoints and reads weather, management and site conditions directly, without PBM output files. The commands below reproduce the recorded experiments.

## Start here

| What to review | File |
|---|---|
| Yield architecture | [daycent_emulator/yield_model.py](daycent_emulator/yield_model.py) |
| SOC architecture and pool loss | [daycent_emulator/soc_model.py](daycent_emulator/soc_model.py) |
| Original files → prepared arrays | [daycent_emulator/raw_data.py](daycent_emulator/raw_data.py) |
| Random site split and preparation | [daycent_emulator/prepare.py](daycent_emulator/prepare.py) |
| Yield features, sampling and paired loss | [daycent_emulator/yield_data.py](daycent_emulator/yield_data.py), [yield_workflow.py](daycent_emulator/yield_workflow.py) |
| SOC features, sampling and rollout | [daycent_emulator/soc_data.py](daycent_emulator/soc_data.py), [soc_workflow.py](daycent_emulator/soc_workflow.py), [soc_loop.py](daycent_emulator/soc_loop.py) |
| Evaluation and R² definitions | [daycent_emulator/evaluate.py](daycent_emulator/evaluate.py), [score_results.py](daycent_emulator/score_results.py), [report.py](daycent_emulator/report.py) |
| Saved results in one table | [results/summary.csv](results/summary.csv) |
| Selected trained checkpoints | [emulator_inference/weights/](emulator_inference/weights/) |
| All five run configurations, exact site IDs and scalers | [runs/](runs/) |
| Frozen defaults | [config.yml](config.yml) |

This is a checkpoint-compatible extraction of the retained final implementation. Model layer names, initialization order, feature calculations and training recipes are preserved. Historical `data/plan.json` files sometimes predate the final architecture; `config.py:effective_plan` applies the recorded final recipe without rewriting those original files.

## Original data and portable setup

`KGML_Midwest_Daily_2026/` is the original supplied Round-3 dataset. `KGML_Midwest_Daily_2026_derived/` was the local prepared-data workspace; another user **does not need that folder**.

Supply the original folder with this layout:

```text
KGML_Midwest_Daily_2026/
  Synthetic/
    InputData/
      Weather.zip
      InitConditions/initial_site_conditions.xlsx
      ScheduleFiles/Synthetic_schedule_management_history_scen1_200.zip
    Output_new_ZippedFolders/batch_*.zip
  Realistic/
    InputFiles/
      Weather.zip
      InitConditions/initial_site_conditions.xlsx
      ScheduleFIles_Realistic/*_management_history.csv
    OutputData/*_harvest.csv
    OutputData/*_monthly.csv
```

The builder reads ZIP members directly. Daily **PBM output** CSVs are not used. Daily weather is required. Monthly and harvest PBM outputs provide training/evaluation targets; model forward passes do not consume their ground-truth labels.

The release reproduces the 198 available scenarios and 9,755 locations in the retained runs. Synthetic scenarios 101–110 were absent from the supplied outputs. The 165 static feature names and their order are pinned in `metadata/static_columns.json` under the Python package. Original schedules use long-form `scenario,id,simyear,doy,management` records; the builder converts these to the same events/N-rate arrays used during training. It does not require the older consolidated-management CSV or `new_data/`.

From this release directory, create an environment outside it:

```bash
python3.12 -m venv ../emulator-env
source ../emulator-env/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

Build a private prepared cache from the original supplied data:

```bash
python run.py prepare-data \
  --raw-root /path/to/KGML_Midwest_Daily_2026 \
  --output /path/to/prepared_data \
  --workers 2
```

The required cache is about **13.2 GiB**, plus temporary management parsing space during construction. Progress is written to `prepared_data/raw_build.json`. Repeating the same command resumes a matching build and checks existing output files. Completion requires all **777** required prepared files to match the retained SHA-256 values exactly. The original supplied files are not modified. No pre-existing derived arrays are copied or read.

Path defaults in `config.yml` are relative to this release: `../KGML_Midwest_Daily_2026` and `../prepared_data`. Set those paths there or use CLI path overrides. Absolute historical paths inside archived run/provenance JSON files identify the original work; `run.py` overrides those paths with the supplied prepared-data root.

## Evaluate the saved models

**Two trained checkpoints are included**, selected by the highest recorded test performance:

| Target | Checkpoint files |
|---|---|
| Yield, site seed 20260915 | `emulator_inference/weights/yield.pt` |
| SOC, site seed 20260914 | `emulator_inference/weights/soc.pt` |

Each checkpoint contains the trained PyTorch state dictionary under `model`. The matching split definitions and normalization statistics are in that run's `data/` directory. The commands below load the saved weights automatically. Other split seeds retain configurations and results for review and retraining, but no weights. Historical checkpoint paths and hashes in their JSON records describe the original runs.

```bash
python run.py evaluate --target yield --site-split-seed 20260915 \
  --input-root /path/to/prepared_data \
  --output /path/to/evaluations/yield_split20260915 --device cuda:0

python run.py evaluate --target soc --site-split-seed 20260914 \
  --input-root /path/to/prepared_data \
  --output /path/to/evaluations/soc_split20260914 --device cuda:0
```

Use a new output directory outside this release. Evaluation copies the selected checkpoint from `emulator_inference/weights/` and its matching archived metadata there, predicts at that run's 975 test sites for all 198 available scenarios, and separately reports the **22 reserved scenarios**. Omitting `--site-split-seed` selects the retained checkpoint for that target. `--device cpu` is supported for evaluation. SOC evaluation includes autoregressive rollout and teacher-forced diagnostics.

Outputs include `predictions/*.npz`, per-scenario metrics, heatmaps, `report/trajectory_metrics.csv`, `report/site_metrics.csv`, and `report/full_period_metrics.json`. Prediction files retain IDs, years, ground truth and predictions. The release's saved scores remain unchanged.

Optional full cache verification, particularly after copying between machines:

```bash
python run.py verify --target soc --input-root /path/to/prepared_data
python run.py verify --target yield --input-root /path/to/prepared_data
```

Evaluation checks input file sizes; the commands above additionally hash every input file.

## Reproduction seeds and split

| Target | Site-split seed | Model/training seed | Training epochs | Selected epoch |
|---|---:|---:|---:|---:|
| Yield, reference | 20260914 | 20261009 | 75 | 75 |
| Yield | 20260915 | 20261009 | 75 | 70 |
| Yield | 20260916 | 20261009 | 75 | 74 |
| SOC, reference | 20260914 | 20261008 | 60 | 58 |
| SOC | 20260915 | 20261008 | 60 | 59 |

The site-split seed changes location membership, **not** the model seed, scenario holdout or time window. For each seed, NumPy `default_rng(seed)` permutes the fixed ordered site list: first 975 test, next 975 validation, remaining 7,805 training. Each membership list is then sorted. Exact IDs are saved in each run's `data/plan.json`; the split generator reproduces them.

- **Training scenarios:** 176 = 170 synthetic + 6 realistic.
- **Reserved scenarios:** 22 = S032, S038, S040, S042, S050, S053, S059, S076, S083, S146, S152, S164, S165, S166, S168, S169, S180, S185, S188, S192, R3, R6.
- **Years:** yield harvests 2001–2023; SOC December states 2000–2023. Year 2000 also supplies yield driver history. These runs have no temporal holdout.
- **Checkpoint selection:** fixed 128-site subset of validation locations and 18 permitted training scenarios. Test locations and reserved scenarios are excluded from optimizer draws and checkpoint selection.
- **Normalization:** fitted on training locations only; saved per-run scalers are reused for evaluation. The original SOC pool/N scaling uses the earlier 20 permitted scenarios; yield label scaling uses all 176 permitted scenarios, separately by crop.

To reproduce a fit from random initialization, explicitly run:

```bash
python run.py train --target yield --site-split-seed 20260915 \
  --input-root /path/to/prepared_data \
  --output /path/to/reproductions/yield_split20260915 --device cuda:0

python run.py train --target soc --site-split-seed 20260915 \
  --input-root /path/to/prepared_data \
  --output /path/to/reproductions/soc_split20260915 --device cuda:0
```

These commands prepare split-specific normalization/sampling, run the original preflight checks, and train the full frozen recipe. They never initialize from the saved best weights. The historical runs used NVIDIA A100-SXM4-40GB GPUs, Python 3.12, PyTorch 2.7.1+cu118, NumPy 2.4.6, TF32 and cuDNN benchmarking. Seeds reproduce splits and initial weights; GPU kernels/hardware can cause small numerical differences in evaluation or retraining.

## Model behavior

**Yield — 626,434 parameters.** A shared 256-hidden-unit LSTM processes the planting-to-harvest sequence, followed by masked attention and one yield head. Inputs are 17 daily features (three weather values, ten practice flags, numeric N rate, two day-of-year encodings, a current-year indicator), 165 static values, a 16-value year encoding, and 64 summaries of up to 730 days of pre-planting weather/management. Crop-specific saved scaling converts the prediction to the source `cgrain` units, gC/m². There is no previous-predicted-yield input.

Training: 75 epochs × 1,024 updates × batch 256; Adam, learning rate 0.001, weight decay 0.0001, gradient clip 5, cosine decay to zero. Anchors sample crops equally and management groups with probabilities 0.30/0.50/0.20 (earlier 20 synthetic / expanded synthetic / realistic). Each anchor is paired with a different allowed scenario at the same site, year and crop. Loss is normalized yield MSE plus 0.5 times the MSE of the paired yield difference.

**SOC — 1,891,760 parameters.** A daily LSTM with 512 hidden units predicts annual changes in four carbon pools. Inputs are 16 daily drivers, the same 165 static values and 16-value year encoding, and four entering pools. The initial pools come from the site workbook. Each predicted December state becomes the next year's entering state. Absolute SOMSC sums the three soil pools; the surface-slow pool is carried but excluded from this sum.

Training: 60 epochs × 512 updates × effective batch 64; Adam, learning rate 0.001, zero weight decay, gradient clip 5, cosine decay to 0.00002. Training uses three eight-year windows per 24-year sample, resetting to true entering states at 2008 and 2016. Annual normalized pool loss includes 0.5 × SOMSC loss; auxiliary January–November supervision has weight 0.5. Those monthly readouts use the full annual driver sequence and are not a monthly autoregressive process.

At evaluation, **autoregressive rollout** starts once from the initial 2000 pools and uses predictions through 2023 without true-state resets. **Teacher forcing** replaces the entering pools with the true prior December state each year and is a diagnostic only. No positivity constraint is added in this locked version: negative pools and accumulated SOC drift remain possible. Yield errors also remain uneven across crop, location and scenario.

## Saved results and meaning of R²

These are recorded scores for 975 held-out sites × 22 reserved scenarios, not newly trained models.

| Yield site seed | Overall pooled R² | Corn pooled R² | Soybean pooled R² | Wheat pooled R² | Median trajectory R²: corn / soybean / wheat |
|---|---:|---:|---:|---:|---|
| 20260914 | 0.98913 | 0.97298 | 0.98190 | 0.92843 | 0.93662 / 0.96886 / 0.87894 |
| 20260915 | 0.98952 | 0.97537 | 0.98374 | 0.92432 | 0.93987 / 0.96878 / 0.87516 |
| 20260916 | 0.98924 | 0.97455 | 0.98391 | 0.92656 | 0.93766 / 0.96997 / 0.87808 |

| SOC site seed, autoregressive | Pooled absolute-stock R² | Median site R² | Median trajectory R² | Final 2023 stock RMSE (gC/m²) |
|---|---:|---:|---:|---:|
| 20260914 | 0.999229 | 0.990079 | 0.979005 | 54.526 |
| 20260915 | 0.998618 | 0.981687 | 0.964732 | 73.109 |

The SOC full-period archive for seed 20260915 was re-evaluated on a GTX 1080 Ti. Its original A100 record is also retained: median trajectory R² 0.964722 and final-stock RMSE 73.105. These are distinct recorded evaluations, not silent rounding replacements.

All R² values use `1 − Σ(predicted − true)² / Σ(true − mean(true))²`, with the mean calculated within the scoring group:

- **Overall pooled:** all valid records across test sites, reserved scenarios and years; yield also pools crops. Per-crop pooled scores use one crop's records.
- **Per-site:** one site's records across years and reserved scenarios; yield is computed separately for each crop. Percentile curves sort these site scores independently by crop/mode.
- **Trajectory:** one site × scenario (× crop for yield), across years. The heatmap headline is the median over these cells. Yield requires at least three harvests and nonzero variance. Undefined cells are excluded from the median; negative R² values are retained.

SOC uses absolute December stocks, with sums accumulated in float64 for scoring. Pooled stock R², site R² and trajectory R² have different denominators and should not be substituted for one another.

## Validation

Runtime outputs are required to be outside the package to preserve the saved models and results. The Python package has no imports from the parent project's implementation.

During packaging, CPU predictions matched the retained source exactly for 12 real yield records spanning all three crops per checkpoint, and two complete 24-year SOC trajectories per checkpoint in both prediction modes. The scoring code matched saved reference metrics for all 975 sites and 22 reserved scenarios. Split memberships, sampling exclusion rules, metric scopes and output protection were also checked before handoff.

The raw builder independently verifies each generated array against `daycent_emulator/metadata/input_manifest.json`. This data manifest is required for preparation and input verification. Full reconstruction from the original supplied data passed: all **777 files were byte-for-byte identical** to the retained inputs, without reading an existing derived folder. It took approximately 31 minutes with three workers on the validation host; the temporary duplicate cache was removed afterward.
