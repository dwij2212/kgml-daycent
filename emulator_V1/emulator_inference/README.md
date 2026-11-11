# Run the selected DayCent yield and SOC emulators on new inputs

This folder can be copied and used on its own. It contains one trained yield
checkpoint, one trained SOC checkpoint, their model code and matching input
normalization. It does not need the parent project's runs, metadata, original
9,755-site dataset or prepared caches.

## Selected weights

Selection follows the requested **highest recorded test performance**: highest
overall pooled yield R² and lowest final-year SOC rollout stock RMSE.

| Output | Saved weights | Source run | Epoch | Recorded test result |
|---|---|---|---:|---|
| Yield | `weights/yield.pt` | `yield_split20260915` | 70 | Pooled R² 0.989516; corn 0.975369, soybean 0.983739, winter wheat 0.924319 |
| SOC | `weights/soc.pt` | `soc_split20260914` | 58 | 2023 stock RMSE 54.526 gC/m²; median trajectory R² 0.979005 |

Those scores use each run's 975 held-out sites and 22 reserved scenarios. The
site partitions differ across repeats, so this is selection from their recorded
results, not a comparison on one common untouched test set. The original
checkpoint files are copied unchanged. `preprocessing.json` contains the
matching normalization and the 165 static feature names, in training order.

## Install

Use Python 3.12. From this folder:

```bash
python3.12 -m venv /path/to/emulator-env
source /path/to/emulator-env/bin/activate
python -m pip install -r requirements.txt
```

The same environment can run CPU inference or CUDA inference. The supplied
requirements match the retained models' environment.

## Supply these three inputs

```text
new_inputs/
  Weather/
    field_A.csv
    field_B.csv
  initial_site_conditions.xlsx
  management.csv
```

Site and scenario IDs are identifiers; new IDs are allowed. A weather filename
must match its site ID exactly. Every site referenced by management must have
one static row and a weather file. Predictions are made for the site/scenario
pairs explicitly present in management.

**Weather:** one CSV per site, with these columns:

```csv
Year,DOY,Tmax,Tmin,Precip
```

Supply every actual calendar day for every requested input year, including
leap days. Use the same weather units as the supplied DayCent dataset; the
loader performs no unit conversion. It reproduces the training model's padded
366-day annual sequences internally. Missing dates and duplicate dates are
rejected.

**Initial site conditions:** XLSX or CSV with `id` and all 165 numeric features
listed in `preprocessing.json` under `static_columns`. Column order in your
file can differ. Missing or non-finite features are rejected; features are not
filled with training averages. Excel uses the `initial_conditions` sheet when
present, otherwise the first sheet.

The four initial carbon pools used by SOC are already among those columns:

| Column | Pool |
|---|---|
| `SOM1CI(2,1)` | Active soil carbon |
| `SOM2CI(2,1)` | Slow soil carbon |
| `SOM3CI(1)` | Passive soil carbon |
| `SOM2CI(1,1)` | Slow surface carbon |

For SOC, these pools must describe the **beginning of `--first-year`**. Subsequent
years use predicted pools. Do not replace them each year with PBM output states.

**Management:** either of the following two CSV formats is supported. Pass
multiple files after `--management` if needed.

1. Long event format, compatible with the original supplied schedules:

   ```csv
   scenario_id,id,Year,doy,management
   new_scenario,field_A,2000,280,winterwheat_planting
   new_scenario,field_A,2001,100,nitrogen_fertilization_1.5gNm2
   new_scenario,field_A,2001,190,harvest_grain
   ```

   Original column names `scenario` and `simyear` are also accepted in place of
   `scenario_id` and `Year`. Use the exact practice names below.

2. Consolidated wide format: identifiers/date plus the ten binary practices
   and nitrogen-rate indicators, for example the earlier handoff header:

   ```csv
   scenario_id,id,Year,doy,conventional_till_molboadplow,corn_planting,cycle_end,harvest_grain,herbicide,nitrogen_fertilization_0gNm2,nitrogen_fertilization_1.5gNm2,nitrogen_fertilization_10.312gNm2,nitrogen_fertilization_17.74gNm2,notill_rodweederrow,reduced_till_tandemdisk,ryegrass_planting,soybean_planting,winterwheat_planting
   ```

   Practice columns contain 0 or 1. Additional nitrogen rates can be represented
   by `nitrogen_fertilization_<rate>gNm2` columns with binary values. The rate in
   the name is the numerical application rate; it is not a category embedding.

Supported non-nitrogen practices:

```text
conventional_till_molboadplow
corn_planting
cycle_end
harvest_grain
herbicide
notill_rodweederrow
reduced_till_tandemdisk
ryegrass_planting
soybean_planting
winterwheat_planting
```

Management is sparse: days without a listed event have no intervention.
Repeated identical events become one event; multiple N rates on the same
site/scenario/day use the maximum rate, matching the original preprocessing.
Dates outside the requested input years are ignored. To represent a completely
unmanaged SOC trajectory, include a zero-N event (or an all-zero wide row) for
that site/scenario in the input interval.

Yield needs an explicit `harvest_grain` event and the preceding planting event.
It predicts corn, soybean and winter wheat, with at most one grain harvest per
site/scenario/year. Ambiguous planting and grain harvest after ryegrass are
rejected. The emulator does not infer future harvest or planting dates.

## Run

Predict both outputs, starting from initial conditions for 2000:

```bash
python predict.py \
  --target both \
  --weather-dir /path/to/new_inputs/Weather \
  --site-conditions /path/to/new_inputs/initial_site_conditions.xlsx \
  --management /path/to/new_inputs/management.csv \
  --first-year 2000 --last-year 2023 \
  --output /path/to/new_predictions \
  --device cpu
```

Use `--device cuda:0 --batch-size 32` for a CUDA GPU. Use `--target yield` or
`--target soc` to run one model. New predictions do not require monthly/harvest
PBM output CSVs or observed yield/SOC values.

**Yield history:** outputs default to `first-year + 1` through `last-year`.
The model uses a planting-to-harvest sequence and up to 730 available days of
drivers before planting. Earlier history is truncated only at `first-year`,
as in training. To obtain full history for a later harvest year, supply three
preceding calendar years, particularly for winter wheat. For example, use
`--first-year 2020 --last-year 2023 --yield-start-year 2023 --target yield` to
emit only 2023 yields. Static values remain the model's initial-condition
features; this history window does not dynamically update them.

**SOC history:** every year from `first-year` through `last-year` is rolled out
in sequence. If initial pools describe 2000, predicting 2023 requires all the
intervening weather and management. Starting at 2023 with the 2000 pools would
describe a different initial-value problem. To simulate a new site or a later
starting year, supply its appropriate initial conditions and drivers.

Year encoding remains anchored to calendar year 2000. The saved model evidence
covers 2000–2023; other years, starting states, climates and management rates
are executable inputs but do not inherit those benchmark accuracy claims.

## Output CSVs

`yield_predictions.csv` contains one row per specified grain harvest in the
yield output interval:

```csv
scenario_id,id,Year,doy,crop,cgrain
```

`cgrain` is predicted grain carbon in the source DayCent units, **gC/m²**. It is
not converted to grain dry matter, tonnes/hectare or bushels/acre.

`soc_predictions.csv` contains one December row per input site/scenario/year:

```csv
scenario_id,id,Year,month,somsc,delta_somsc,som1c_soil,som2c_soil,som3c,som2c_surface
```

Pool values and SOMSC are in **gC/m²**. `somsc` is the sum of the first three
soil pools; the surface pool is carried separately. `delta_somsc` is the change
from the entering state (initial pools in the first year, previous predicted
December thereafter). These outputs are annual; the model's auxiliary monthly
training head is not exported as a monthly forecast. No pool clipping or bias
correction is applied, so the selected model's negative-pool and drift behavior
is preserved.

Existing prediction CSVs are not overwritten. Use a new output directory for
another inference call. Output rows carry site/scenario/year identifiers; SOC
rows are streamed in batches and can be sorted by those columns if desired.

The input-only path was checked against the selected saved implementations on
92 harvests and four complete 24-year SOC trajectories. Input features matched
exactly and predictions agreed to floating-point precision. A standalone copy
also ran with renamed site IDs, both management formats were checked, and
missing static features/weather dates were rejected. These checks verify the
inference implementation; the performance scores above remain the recorded
experiment results.
