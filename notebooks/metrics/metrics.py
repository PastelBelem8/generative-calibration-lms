from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

import logging

import pandas as pd
import numpy as np
import scipy.stats as st

from . import f_metrics as f
from . import hf_metrics as hf
from . import calibration_metrics as calib


INT_CONST = 10**6


@dataclass
class BaseMetric(ABC):
    target_label: str
    pred_label: str

    metrics: dict = field(default_factory=dict)

    def _run_metric(self, metric_name, metric_fn, *args, **kwargs):
        """Auxiliary function to avoid placeholder code"""
        results = {}
        result = metric_fn(*args, **kwargs)

        if isinstance(result, dict):
            results.update(result)
        else:
            results[metric_name] = result
        return results

    @abstractmethod
    def compute(self, data):
        raise NotImplementedError("Must be overriden by subclasses")


@dataclass
class RowWiseMetric(BaseMetric, ABC):
    @abstractmethod
    def _compute_per_row(self, args):
        raise NotImplementedError("Must be overriden by subclasses")

    def compute(self, data):
        return data.apply(self._compute_per_row, axis=1, result_type="expand")


@dataclass
class PerformanceMetrics(RowWiseMetric):
    target_multi_label: str = None

    token_suffix: str = "_token"

    hugface_kwargs: str = field(default_factory=dict)

    metrics_token: dict = field(default_factory=dict)
    metrics_text: dict = field(default_factory=dict)

    metrics_multi_token: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.hugface_kwargs:
            self.hugface_kwargs = {
                "keep_in_memory": True,
                "seed": np.random.randint(INT_CONST),
            }

        if not self.metrics_token:
            self.metrics_token = {
                "exact_match": f.exact_match,
                "first_error_position": f.first_error_position,
                "f_metrics": f.f_metrics,
            }

        if not self.metrics_text:
            self.metrics_text = {
                "rouge": hf.hf_metric(
                    "rouge", hf.parse_hf_rouge, **self.hugface_kwargs
                ),
                "meteor": hf.hf_metric("meteor", **self.hugface_kwargs),
            }

        if self.target_multi_label and not self.metrics_multi_token:
            self.metrics_multi_token = {
                "bleu": hf.hf_metric("bleu", hf.parse_hf_bleu, **self.hugface_kwargs),
            }

    def token_colname(self, col: str) -> str:
        return col + self.token_suffix

    def _compute_per_row(self, args):
        # Metrics values for row
        metrics = {}

        target_label_token = self.token_colname(self.target_label)
        pred_label_token = self.token_colname(self.pred_label)

        y_true_tokens = args[target_label_token]
        y_pred_tokens = args[pred_label_token]

        # ------------------------------
        # Token metrics
        # ------------------------------
        tokens_kwargs = {"y_true": y_true_tokens, "y_pred": y_pred_tokens}
        for metric_name, metric_fn in self.metrics_token.items():
            rs = self._run_metric(metric_name, metric_fn, **tokens_kwargs)
            metrics.update(rs)
        del tokens_kwargs

        # --------------------------------------------------------------
        # Hugging face metrics
        # --------------------------------------------------------------
        # 1. Text metrics
        y_true_text = " ".join(y_true_tokens)
        y_pred_text = " ".join(y_pred_tokens)
        hf_kwargs = {"predictions": [y_pred_text], "references": [y_true_text]}

        for metric_name, metric_fn in self.metrics_text.items():
            rs = self._run_metric(metric_name, metric_fn, **hf_kwargs)
            metrics.update(rs)

        # 2. Multi-label
        if self.metrics_multi_token:
            target_label_token: str = self.token_colname(self.target_multi_label)

            token_kwargs = {
                "predictions": [y_pred_tokens],
                "references": [args[target_label_token]],
            }
            for metric_name, metric_fn in self.metrics_multi_token.items():
                rs = self._run_metric(metric_name, metric_fn, **token_kwargs)
                metrics.update(rs)
            del token_kwargs

        metrics["metric_type"] = "performance"
        return metrics


@dataclass
class CorrelationMetric:
    reference_label: str
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.metrics:
            self.metrics = {
                "pearsonr": st.pearsonr,
                "spearmanr": st.spearmanr,
                "kendall_tau": st.kendalltau,
            }

    def compute(self, data, cols):
        # 1. Initialization
        if isinstance(cols, str):
            cols = [cols]

        # 2. Validation
        # We only consider columns without nans
        new_cols = []
        for col in cols:
            if data[col].isna().any():
                logging.warning(f"Skipping computation for col {col} due to NaN")
                continue
            elif col == self.reference_label:
                continue
            new_cols.append(col)

        # 3. Execute
        y_ref = data[self.reference_label]

        results = defaultdict(list)
        results["x"].extend(new_cols)
        results["y"].extend([self.reference_label] * len(new_cols))

        for metric_name, metric_fn in self.metrics.items():
            corrs_pvals = [metric_fn(y_ref, data[c]) for c in new_cols]
            corrs, pvals = zip(*corrs_pvals)
            results[metric_name].extend(corrs)
            results[f"{metric_name}_pvalue"].extend(pvals)

        results = pd.DataFrame(results)
        results["metric_type"] = "correlation"
        return results


@dataclass
class CalibrationMetrics:
    reference_label: str
    n_bins: int = 20
    frac: float = 0.1

    _calib_errors: list = field(init=False, default_factory=dict)
    _eq_width_errors: list = field(init=False, default_factory=dict)
    _eq_freq_errors: list = field(init=False, default_factory=dict)

    def compute(self, data, cols):
        if isinstance(cols, str):
            cols = [cols]

        y_ref = data[self.reference_label].values

        results = defaultdict(list)
        for col in cols:
            y_other = data[col].values

            results["x"].append(self.reference_label)
            results["y"].append(col)

            mse = np.mean(calib.sq_error(y_ref, y_other))
            results["mse"].append(mse)

            self._calib_errors[col] = calib.calibration_error(y_ref, y_other)
            mae = np.mean(np.abs(self._calib_errors[col]))
            results["mae"].append(mae)

            results["ce_avg"].append(np.mean(self._calib_errors[col]))
            results["ce_std"].append(np.std(self._calib_errors[col]))

            self._eq_width_errors[col] = calib.equal_width_ece_bins(
                y_other, y_ref, self.n_bins
            )
            results["ECE_eq_width"].append(np.sum(self._eq_width_errors[col]))
            results[f"ECE_eq_width_max"].append(np.max(self._eq_width_errors[col]))

            self._eq_freq_errors[col] = calib.equal_freq_ece(y_other, y_ref, self.frac)
            results["ECE_eq_freq"].append(np.sum(self._eq_freq_errors[col]))
            results[f"ECE_eq_freq_max"].append(np.max(self._eq_freq_errors[col]))
            results["hyperparams"].append({"n_bins": self.n_bins, "frac": self.frac})

        results = pd.DataFrame(results)
        results["metric_type"] = "calibration"
        return results
