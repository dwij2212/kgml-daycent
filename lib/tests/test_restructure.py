"""
Rigorous test suite for the kgml-daycent restructure.

Tests cover:
  1. utils.metrics  — compute_regression_metrics, compute_masked_metrics,
                      compute_per_channel_metrics, compute_emulator_metrics
  2. utils.plotting — every public function produces a Figure without error
  3. utils.config   — ExperimentConfig.from_yaml, InverseExperimentConfig
                      dataclass construction, InverseDataConfig.resolve_points
  4. utils.__init__ — all expected symbols importable from the top-level package
  5. selection       — MCDropoutAcquisition / LatentDiversityAcquisition registry
                       names, RandomAcquisition still works
  6. data.dataset    — DayCentDataset removed; DayCentDatasetV2 still importable
  7. data.preprocessing — split_by_quadrants / load_data removed;
                          active functions still importable
  8. eval_emulator   — imports cleanly (no plt / old calculate_metrics references)
  9. eval_inverse    — imports cleanly (no mean_squared_error / r2_score references)
 10. train_inverse   — imports InverseExperimentConfig from utils.config (not inverse_config)
 11. Dead-file guard — utils/sampler.py, utils/inverse_config.py,
                       test_dataset_loader.py do NOT exist
 12. Config paths    — flat configs removed; subdirectory configs present
 13. Script paths    — flat scripts removed; subdirectory scripts present
"""

import importlib
import inspect
import os
import sys
import types
import unittest
import numpy as np
import tempfile
import yaml

# Make sure lib/ is on the path regardless of where pytest is invoked from
LIB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI


# ============================================================
# 1. utils.metrics
# ============================================================
class TestMetrics(unittest.TestCase):

    def _dummy(self, n=100, c=4):
        rng = np.random.default_rng(0)
        true = rng.normal(0, 1, (n, c)).astype(np.float32)
        pred = true + rng.normal(0, 0.1, (n, c)).astype(np.float32)
        return true, pred

    def test_import(self):
        from utils.metrics import (
            compute_regression_metrics,
            compute_masked_metrics,
            compute_per_channel_metrics,
            compute_emulator_metrics,
        )

    def test_compute_regression_metrics_keys(self):
        from utils.metrics import compute_regression_metrics
        true = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.1, 1.9, 3.1, 3.9])
        m = compute_regression_metrics(true, pred)
        for key in ("mse", "rmse", "mae", "r2", "n_samples"):
            self.assertIn(key, m, f"Missing key '{key}'")
        self.assertEqual(m["n_samples"], 4)

    def test_regression_perfect_prediction(self):
        from utils.metrics import compute_regression_metrics
        arr = np.arange(1, 11, dtype=float)
        m = compute_regression_metrics(arr, arr)
        self.assertAlmostEqual(m["mse"], 0.0, places=8)
        self.assertAlmostEqual(m["r2"],  1.0, places=8)

    def test_compute_masked_metrics_filters(self):
        from utils.metrics import compute_masked_metrics
        true = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 2.0, 999., 999.])   # last two should be masked out
        mask = np.array([1, 1, 0, 0], dtype=float)
        m = compute_masked_metrics(true, pred, mask)
        self.assertEqual(m["n_samples"], 2)
        self.assertAlmostEqual(m["mse"], 0.0, places=8)

    def test_compute_per_channel_metrics_shape(self):
        from utils.metrics import compute_per_channel_metrics
        true, pred = self._dummy(n=50, c=6)
        names = [f"feat_{i}" for i in range(6)]
        results = compute_per_channel_metrics(true, pred, channel_names=names)
        self.assertEqual(len(results), 6)
        for r in results:
            for key in ("channel_idx", "channel_name", "mse", "r2", "corr"):
                self.assertIn(key, r)

    def test_compute_per_channel_metrics_filter(self):
        from utils.metrics import compute_per_channel_metrics
        true, pred = self._dummy(n=50, c=6)
        names = ["alpha", "beta", "gamma", "alpha_2", "delta", "beta_sub"]
        results = compute_per_channel_metrics(
            true, pred, channel_names=names, filter_names=["alpha", "beta"]
        )
        # Should only return channels whose name starts with 'alpha' or 'beta'
        returned_names = [r["channel_name"] for r in results]
        self.assertEqual(set(returned_names), {"alpha", "beta", "alpha_2", "beta_sub"})

    def test_compute_emulator_metrics_structure(self):
        from utils.metrics import compute_emulator_metrics
        rng = np.random.default_rng(1)
        N = 60
        yield_true = rng.normal(500, 50, N).astype(np.float32)
        yield_pred = yield_true + rng.normal(0, 5, N).astype(np.float32)
        somsc_true = rng.normal(200, 20, (N, 12)).astype(np.float32)
        somsc_pred = somsc_true + rng.normal(0, 2, (N, 12)).astype(np.float32)

        preds = {
            "yield_true":  yield_true,
            "yield_pred":  yield_pred,
            "yield_mask":  np.ones(N, dtype=float),
            "somsc_true":  somsc_true,
            "somsc_pred":  somsc_pred,
            "somsc_mask":  np.ones((N, 12), dtype=float),
        }
        m = compute_emulator_metrics(preds)
        self.assertIn("yield", m)
        self.assertIn("somsc", m)
        for task in ("yield", "somsc"):
            for key in ("mse", "rmse", "mae", "r2"):
                self.assertIn(key, m[task])

    def test_emulator_metrics_with_nan_mask(self):
        """Masked-out samples must not affect metrics."""
        from utils.metrics import compute_emulator_metrics
        rng = np.random.default_rng(2)
        N = 40
        yield_true = rng.normal(0, 1, N).astype(np.float32)
        yield_pred = yield_true.copy()                          # perfect prediction
        # mask only first 20 as valid
        yield_mask = np.zeros(N, dtype=float)
        yield_mask[:20] = 1.0
        # corrupt the masked-out values — should be ignored
        yield_pred[20:] = 9999.0

        somsc_true = rng.normal(0, 1, (N, 12)).astype(np.float32)
        somsc_pred = somsc_true.copy()
        somsc_mask = np.ones((N, 12), dtype=float)

        preds = {
            "yield_true": yield_true, "yield_pred": yield_pred,
            "yield_mask": yield_mask,
            "somsc_true": somsc_true, "somsc_pred": somsc_pred,
            "somsc_mask": somsc_mask,
        }
        m = compute_emulator_metrics(preds)
        self.assertAlmostEqual(m["yield"]["mse"], 0.0, places=4)


# ============================================================
# 2. utils.plotting
# ============================================================
class TestPlotting(unittest.TestCase):

    def setUp(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        rng = np.random.default_rng(3)
        self.true = rng.normal(0, 1, 80).astype(np.float32)
        self.pred = self.true + rng.normal(0, 0.2, 80).astype(np.float32)

    def _is_fig(self, obj):
        import matplotlib.pyplot as plt
        self.assertIsInstance(obj, self.plt.Figure)
        self.plt.close("all")

    def test_plot_timeseries(self):
        from utils.plotting import plot_timeseries
        fig = plot_timeseries(self.true, self.pred, title="test",
                              metrics={"r2": 0.95})
        self._is_fig(fig)

    def test_plot_dual_timeseries(self):
        from utils.plotting import plot_dual_timeseries
        fig = plot_dual_timeseries(
            series=[(self.true, self.pred, "Yield"),
                    (self.true * 2, self.pred * 2, "SOMSC")],
            titles=["Yield", "SOMSC"],
        )
        self._is_fig(fig)

    def test_plot_scatter(self):
        from utils.plotting import plot_scatter
        fig = plot_scatter(self.true, self.pred, title="scatter test")
        self._is_fig(fig)

    def test_plot_scatter_grid(self):
        from utils.plotting import plot_scatter_grid
        rng = np.random.default_rng(4)
        true_2d = rng.normal(0, 1, (80, 5)).astype(np.float32)
        pred_2d = true_2d + rng.normal(0, 0.1, (80, 5)).astype(np.float32)
        results = [
            {"channel_idx": i, "channel_name": f"feat_{i}",
             "corr": 0.9, "r2": 0.8}
            for i in range(5)
        ]
        fig = plot_scatter_grid(results, true_2d, pred_2d, ncols=3)
        self._is_fig(fig)

    def test_plot_bar_h(self):
        from utils.plotting import plot_bar_h
        fig = plot_bar_h(
            names=["a", "b", "c"],
            values=[0.5, 0.8, 0.3],
            title="R² bar",
            xlabel="R²",
        )
        self._is_fig(fig)

    def test_plot_budget_curve(self):
        from utils.plotting import plot_budget_curve
        fig = plot_budget_curve(
            n_points=[10, 20, 30, 40],
            metrics_by_task={
                "yield": {"r2": [0.5, 0.6, 0.7, 0.75],
                          "rmse": [10, 9, 8, 7]},
                "somsc": {"r2": [0.4, 0.55, 0.65, 0.7]},
            },
        )
        self._is_fig(fig)

    def test_plot_distribution_comparison(self):
        from utils.plotting import plot_distribution_comparison
        rng = np.random.default_rng(5)
        pool = {f: rng.normal(0, 1, 200) for f in ["sand", "clay", "ph"]}
        sel  = {f: rng.normal(0, 1,  50) for f in ["sand", "clay", "ph"]}
        fig = plot_distribution_comparison(pool, sel, title="Coverage")
        self._is_fig(fig)

    def test_plot_spatial_points_saves(self):
        from utils.plotting import plot_spatial_points
        rng = np.random.default_rng(6)
        N = 100
        lon = rng.uniform(-100, -80, N)
        lat = rng.uniform(35, 48, N)
        elev = rng.uniform(100, 500, N)
        test_mask  = np.zeros(N, dtype=bool); test_mask[:10] = True
        train_mask = np.zeros(N, dtype=bool); train_mask[20:40] = True
        fig = plot_spatial_points(lon, lat, elev, train_mask, test_mask)
        self._is_fig(fig)

    def test_save_path_creates_dir(self):
        from utils.plotting import plot_timeseries
        with tempfile.TemporaryDirectory() as td:
            save_path = os.path.join(td, "subdir", "test.png")
            plot_timeseries(self.true, self.pred, save_path=save_path)
            self.assertTrue(os.path.exists(save_path))
        self.plt.close("all")


# ============================================================
# 3. utils.config
# ============================================================
class TestConfig(unittest.TestCase):

    def test_experiment_config_import(self):
        from utils.config import (
            SplitConfig, DataConfig, ModelConfig,
            TrainingConfig, WandbConfig, ExperimentConfig,
        )

    def test_inverse_config_in_utils_config(self):
        """InverseExperimentConfig must live in utils.config now."""
        from utils.config import (
            InverseDataConfig, InverseTrainingConfig, InverseExperimentConfig,
        )

    def test_experiment_config_from_yaml(self):
        yaml_path = os.path.join(LIB_DIR, "configs/emulator/experiment_1.yaml")
        if not os.path.exists(yaml_path):
            self.skipTest(f"Config not found: {yaml_path}")
        from utils.config import ExperimentConfig
        cfg = ExperimentConfig.from_yaml(yaml_path)
        self.assertIsNotNone(cfg.experiment_id)

    def test_inverse_data_config_resolve_points_explicit(self):
        from utils.config import InverseDataConfig
        dc = InverseDataConfig(
            points_train=["1", "2", "3"],
            points_val=["4"],
            points_test=["5"],
        )
        dc.resolve_points(seed=0)
        self.assertEqual(dc._train_points, ["1", "2", "3"])
        self.assertEqual(dc._val_points, ["4"])
        self.assertEqual(dc._test_points, ["5"])

    def test_inverse_experiment_config_dirs_created(self):
        from utils.config import InverseExperimentConfig, InverseDataConfig, InverseTrainingConfig
        with tempfile.TemporaryDirectory() as td:
            cfg = InverseExperimentConfig(
                experiment_id="test_inv",
                data=InverseDataConfig(
                    points_train=["1"], points_val=["2"], points_test=["3"]
                ),
                output_dir=td,
            )
            cfg.data.resolve_points()
            self.assertTrue(os.path.isdir(cfg.model_dir))
            self.assertTrue(os.path.isdir(cfg.result_dir))

    def test_inverse_config_not_importable_from_old_path(self):
        """utils.inverse_config must NOT exist after deletion."""
        with self.assertRaises((ImportError, ModuleNotFoundError)):
            import utils.inverse_config  # noqa: F401


# ============================================================
# 4. utils.__init__  —  all expected symbols present
# ============================================================
class TestUtilsInit(unittest.TestCase):

    EXPECTED = [
        # eval
        "evaluate",
        # config — emulator
        "SplitConfig", "DataConfig", "ModelConfig",
        "TrainingConfig", "WandbConfig", "ExperimentConfig",
        # config — inverse
        "InverseDataConfig", "InverseTrainingConfig", "InverseExperimentConfig",
        # metrics
        "compute_regression_metrics", "compute_masked_metrics",
        "compute_per_channel_metrics", "compute_emulator_metrics",
        # plotting
        "plot_timeseries", "plot_dual_timeseries",
        "plot_scatter", "plot_scatter_grid",
        "plot_bar_h", "plot_budget_curve",
        "plot_distribution_comparison", "plot_spatial_points",
        # optim
        "build_optimizer", "build_scheduler",
        "create_optimizer_and_scheduler",
        # training
        "setup_reproducibility", "get_device",
        "CheckpointManager", "WandbLogger",
    ]

    REMOVED = ["ScenarioWiseSampler"]

    def test_all_expected_present(self):
        import utils
        for name in self.EXPECTED:
            self.assertTrue(
                hasattr(utils, name),
                f"utils.{name} is missing from __init__.py",
            )

    def test_removed_symbols_absent(self):
        import utils
        for name in self.REMOVED:
            self.assertFalse(
                hasattr(utils, name),
                f"utils.{name} should have been removed but is still present",
            )


# ============================================================
# 5. selection package
# ============================================================
class TestSelection(unittest.TestCase):

    def test_mc_dropout_registered(self):
        from selection.acquisition import ACQUISITION_REGISTRY
        self.assertIn("mc_dropout", ACQUISITION_REGISTRY)
        self.assertNotIn("uncertainty", ACQUISITION_REGISTRY)

    def test_latent_diversity_registered(self):
        from selection.acquisition import ACQUISITION_REGISTRY
        self.assertIn("latent_diversity", ACQUISITION_REGISTRY)
        self.assertNotIn("diversity", ACQUISITION_REGISTRY)

    def test_random_acquisition_works(self):
        from selection import get_acquisition
        acq = get_acquisition("random", seed=99)
        scores = acq.score("dummy.pth", ["p1", "p2", "p3"], [], {})
        self.assertEqual(set(scores.keys()), {"p1", "p2", "p3"})
        for v in scores.values():
            self.assertIsInstance(v, float)

    def test_mc_dropout_raises_not_implemented(self):
        from selection import get_acquisition
        acq = get_acquisition("mc_dropout")
        with self.assertRaises(NotImplementedError):
            acq.score("m.pth", ["p1"], [], {})

    def test_latent_diversity_raises_not_implemented(self):
        from selection import get_acquisition
        acq = get_acquisition("latent_diversity")
        with self.assertRaises(NotImplementedError):
            acq.score("m.pth", ["p1"], [], {})

    def test_select_top(self):
        from selection import get_acquisition
        acq = get_acquisition("random", seed=0)
        scores = {"a": 0.9, "b": 0.2, "c": 0.5, "d": 0.8}
        top2 = acq.select_top(scores, n=2)
        self.assertEqual(set(top2), {"a", "d"})

    def test_init_exports(self):
        from selection import (
            MCDropoutAcquisition, LatentDiversityAcquisition,
            RandomAcquisition, BaseAcquisition,
            get_acquisition, get_strategy,
        )

    def test_old_names_not_in_init(self):
        import selection
        self.assertFalse(hasattr(selection, "UncertaintyAcquisition"),
                         "UncertaintyAcquisition should be gone from selection.__init__")
        self.assertFalse(hasattr(selection, "DiversityAcquisition"),
                         "DiversityAcquisition should be gone from selection.__init__")


# ============================================================
# 6. data.dataset  —  DayCentDataset removed, V2 present
# ============================================================
class TestDataset(unittest.TestCase):

    def test_daycent_dataset_v2_importable(self):
        from data.dataset import DayCentDatasetV2
        self.assertTrue(inspect.isclass(DayCentDatasetV2))

    def test_legacy_dataset_removed(self):
        from data import dataset as ds_module
        self.assertFalse(
            hasattr(ds_module, "DayCentDataset"),
            "Legacy DayCentDataset should have been removed from data/dataset.py",
        )


# ============================================================
# 7. data.preprocessing  —  dead functions removed, live ones present
# ============================================================
class TestPreprocessing(unittest.TestCase):

    def test_dead_functions_removed(self):
        from data import preprocessing as pp
        self.assertFalse(hasattr(pp, "split_by_quadrants"),
                         "split_by_quadrants should have been removed")
        self.assertFalse(hasattr(pp, "load_data"),
                         "load_data should have been removed")

    def test_live_functions_present(self):
        from data.preprocessing import (
            load_output_data,
            load_management_data,
            normalize_outputs,
            normalize_weather_data,
            prepare_data_for_datasetv2,
        )


# ============================================================
# 8. eval_emulator  —  clean import, correct function names
# ============================================================
class TestEvalEmulator(unittest.TestCase):

    def test_module_imports_cleanly(self):
        """eval_emulator must import without errors (no plt at module level)."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "eval_emulator",
            os.path.join(LIB_DIR, "eval_emulator.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass  # argparse may call sys.exit on import with no args — that's fine

    def test_calculate_metrics_delegates(self):
        """calculate_metrics in eval_emulator should call compute_emulator_metrics."""
        import importlib, ast
        src = open(os.path.join(LIB_DIR, "eval_emulator.py")).read()
        # The hand-rolled loops must be gone
        self.assertNotIn("yield_mse = np.mean", src,
                         "Hand-rolled yield MSE found — should use utils.metrics")
        self.assertNotIn("somsc_mse = np.mean", src,
                         "Hand-rolled SOMSC MSE found — should use utils.metrics")
        # The delegation call must be present
        self.assertIn("compute_emulator_metrics", src)

    def test_no_raw_plt_import_at_top(self):
        src = open(os.path.join(LIB_DIR, "eval_emulator.py")).read()
        # plt must NOT be imported as a top-level standalone import
        # (it's fine inside function bodies)
        import_lines = [l for l in src.splitlines()
                        if l.startswith("import matplotlib") or
                           l.startswith("import matplotlib.pyplot")]
        self.assertEqual(import_lines, [],
                         f"Unexpected top-level matplotlib import: {import_lines}")


# ============================================================
# 9. eval_inverse  —  no hand-rolled sklearn metrics
# ============================================================
class TestEvalInverse(unittest.TestCase):

    def test_no_sklearn_metrics_import(self):
        src = open(os.path.join(LIB_DIR, "eval_inverse.py")).read()
        self.assertNotIn("from sklearn.metrics import",
                         src,
                         "sklearn.metrics should no longer be imported directly in eval_inverse.py")

    def test_uses_compute_per_channel_metrics(self):
        src = open(os.path.join(LIB_DIR, "eval_inverse.py")).read()
        self.assertIn("compute_per_channel_metrics", src)

    def test_uses_utils_plotting(self):
        src = open(os.path.join(LIB_DIR, "eval_inverse.py")).read()
        self.assertIn("from utils.plotting import", src)


# ============================================================
# 10. train_inverse  —  imports from utils.config
# ============================================================
class TestTrainInverse(unittest.TestCase):

    def test_imports_from_utils_config(self):
        src = open(os.path.join(LIB_DIR, "train_inverse.py")).read()
        self.assertIn("from utils.config import InverseExperimentConfig", src)

    def test_no_inverse_config_import(self):
        src = open(os.path.join(LIB_DIR, "train_inverse.py")).read()
        self.assertNotIn("utils.inverse_config", src)
        self.assertNotIn("inverse_config import", src)


# ============================================================
# 11. Dead-file guard
# ============================================================
class TestDeadFilesGone(unittest.TestCase):

    def _absent(self, rel_path):
        full = os.path.join(LIB_DIR, rel_path)
        self.assertFalse(
            os.path.exists(full),
            f"Expected deleted file still exists: {full}",
        )

    def test_sampler_deleted(self):
        self._absent("utils/sampler.py")

    def test_inverse_config_deleted(self):
        self._absent("utils/inverse_config.py")

    def test_test_dataset_loader_deleted(self):
        self._absent("test_dataset_loader.py")

    def test_train_experiment_deleted(self):
        self._absent("train_experiment.py")

    def test_eval_experiment_deleted(self):
        self._absent("eval_experiment.py")


# ============================================================
# 12. Config directory structure
# ============================================================
class TestConfigPaths(unittest.TestCase):

    def _exists(self, rel_path):
        full = os.path.join(LIB_DIR, rel_path)
        self.assertTrue(os.path.exists(full), f"Expected file missing: {full}")

    def _absent(self, rel_path):
        full = os.path.join(LIB_DIR, rel_path)
        self.assertFalse(os.path.exists(full), f"Old flat file still exists: {full}")

    def test_emulator_configs_exist(self):
        for name in ["experiment_1.yaml", "experiment_2.yaml",
                     "experiment_3.yaml", "experiment_4.yaml",
                     "experiment18.yaml", "legacy_experiment.yaml",
                     "transformer.yaml"]:
            self._exists(f"configs/emulator/{name}")

    def test_inverse_config_exists(self):
        self._exists("configs/inverse/inverse_1.yaml")

    def test_selection_config_exists(self):
        self._exists("configs/selection/selection_base.yaml")

    def test_flat_configs_gone(self):
        for name in ["experiment_1.yaml", "experiment_2.yaml",
                     "inverse_1.yaml", "selection_base.yaml"]:
            self._absent(f"configs/{name}")


# ============================================================
# 13. Script directory structure
# ============================================================
class TestScriptPaths(unittest.TestCase):

    def _exists(self, rel_path):
        full = os.path.join(LIB_DIR, rel_path)
        self.assertTrue(os.path.exists(full), f"Expected script missing: {full}")

    def _absent(self, rel_path):
        full = os.path.join(LIB_DIR, rel_path)
        self.assertFalse(os.path.exists(full), f"Old flat script still exists: {full}")

    def test_emulator_scripts_exist(self):
        for name in ["train.sh", "eval.sh", "train_legacy.sh", "eval_legacy.sh"]:
            self._exists(f"scripts/emulator/{name}")

    def test_inverse_script_exists(self):
        self._exists("scripts/inverse/train_eval.sh")

    def test_selection_scripts_exist(self):
        for name in ["run_random_sweep.sh", "run_stratified_sweep.sh",
                     "run_random_and_stratified.sh", "compare_strategies.sh"]:
            self._exists(f"scripts/selection/{name}")

    def test_flat_scripts_gone(self):
        for name in ["train.sh", "eval.sh", "train_legacy.sh", "eval_legacy.sh",
                     "run_random_sweep.sh", "run_stratified_sweep.sh",
                     "compare_strategies.sh"]:
            self._absent(f"scripts/{name}")

    def test_emulator_scripts_reference_new_entrypoints(self):
        for sh in ["train.sh", "eval.sh"]:
            src = open(os.path.join(LIB_DIR, f"scripts/emulator/{sh}")).read()
            self.assertIn("train_emulator.py" if "train" in sh else "eval_emulator.py", src)
            self.assertNotIn("train_experiment.py", src)
            self.assertNotIn("eval_experiment.py", src)

    def test_emulator_scripts_reference_new_config_dir(self):
        for sh in ["train.sh", "eval.sh"]:
            src = open(os.path.join(LIB_DIR, f"scripts/emulator/{sh}")).read()
            self.assertIn("configs/emulator/", src)
            # Old flat path must be gone
            self.assertNotIn("configs/experiment_", src)

    def test_selection_scripts_reference_new_config_dir(self):
        for sh in ["run_random_sweep.sh", "run_stratified_sweep.sh"]:
            src = open(os.path.join(LIB_DIR, f"scripts/selection/{sh}")).read()
            self.assertIn("configs/selection/selection_base.yaml", src)


# ============================================================
# 14. run_*_experiment.py — updated imports
# ============================================================
class TestRunScripts(unittest.TestCase):

    def test_run_selection_uses_new_entrypoints(self):
        src = open(os.path.join(LIB_DIR, "run_selection_experiment.py")).read()
        self.assertIn("from train_emulator import", src)
        self.assertIn("from eval_emulator import", src)
        self.assertNotIn("from train_experiment import", src)
        self.assertNotIn("from eval_experiment import", src)

    def test_run_active_uses_new_entrypoints(self):
        src = open(os.path.join(LIB_DIR, "run_active_experiment.py")).read()
        self.assertIn("from train_emulator import", src)
        self.assertIn("from eval_emulator import", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
