"""
LightGBM Direct 12-Month-Sum Forecasting Pipeline (Comprehensive)
===================================================================
Input: DataFrame with columns [tin, ds, y]
  - tin: identifier for each time series
  - ds:  date (monthly period)
  - y:   monthly cashflow amount

Output: 12-month-sum forecast per tin, evaluated with median APE by group.

Feature categories:
  1.  Tenure
  2.  Trailing aggregates (raw + normalized)
  3.  Ratios between windows
  4.  Year-over-year
  5.  Trend (slope, momentum, acceleration)
  6.  Diff / change features
  7.  Seasonality
  8.  Volatility / distribution shape
  9.  Outlier detection & characterization
  10. Sparsity / zero patterns
  11. Structural break / regime change
  12. Recency & level-shift
  13. Percentile / rank features
  14. Concentration / Gini
  15. Log-scale features
  16. Autocorrelation
  17. Rolling volatility change
  18. Extreme value features
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and sort the input dataframe."""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["tin", "ds"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["tin", "ds"], keep="last")
    return df


def fill_missing_months(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps with 0 so every tin has contiguous monthly data."""
    records = []
    for tin, grp in df.groupby("tin"):
        grp = grp.set_index("ds").sort_index()
        full_range = pd.date_range(grp.index.min(), grp.index.max(), freq="MS")
        grp = grp.reindex(full_range, fill_value=0)
        grp["tin"] = tin
        grp.index.name = "ds"
        records.append(grp.reset_index())
    return pd.concat(records, ignore_index=True)


# =============================================================================
# HELPER
# =============================================================================

def _safe_window(history: np.ndarray, w: int) -> np.ndarray:
    """Return last w elements, or full history if shorter."""
    return history[-w:] if len(history) >= w else history


# =============================================================================
# 2. COMPREHENSIVE FEATURE ENGINEERING
# =============================================================================

def make_features_for_tin(
    cashflows: np.ndarray,
    origin_idx: int,
    origin_date: pd.Timestamp,
) -> dict:
    """
    Build comprehensive features for a single tin at a single origin point.
    """
    history = cashflows[: origin_idx + 1].astype(float)
    tenure = len(history)
    eps = 1e-9
    feats = {}

    # ==================================================================
    # CATEGORY 1: TENURE
    # ==================================================================
    feats["tenure"] = tenure
    feats["tenure_bucket"] = min(tenure // 6, 8)
    feats["tenure_log"] = np.log1p(tenure)

    # ==================================================================
    # CATEGORY 2: TRAILING AGGREGATES (raw + pct-based)
    # ==================================================================
    for w in [1, 2, 3, 6, 9, 12, 18, 24]:
        win = _safe_window(history, w)
        feats[f"sum_{w}m"] = win.sum()
        feats[f"mean_{w}m"] = win.mean()
        feats[f"std_{w}m"] = win.std()
        feats[f"min_{w}m"] = win.min()
        feats[f"max_{w}m"] = win.max()
        feats[f"median_{w}m"] = np.median(win)
        feats[f"has_{w}m"] = int(tenure >= w)
        feats[f"range_pct_{w}m"] = (win.max() - win.min()) / (win.mean() + eps)

    feats["annualized_rate"] = history.mean() * 12
    feats["lifetime_sum"] = history.sum()
    feats["lifetime_mean"] = history.mean()
    feats["lifetime_std"] = history.std()
    feats["lifetime_median"] = np.median(history)

    # ==================================================================
    # CATEGORY 3: RATIOS BETWEEN WINDOWS
    # ==================================================================
    feats["ratio_1m_3m"] = feats["sum_1m"] / (feats["sum_3m"] + eps)
    feats["ratio_1m_6m"] = feats["sum_1m"] / (feats["sum_6m"] + eps)
    feats["ratio_1m_12m"] = feats["sum_1m"] / (feats["sum_12m"] + eps)
    feats["ratio_3m_6m"] = feats["sum_3m"] / (feats["sum_6m"] + eps)
    feats["ratio_3m_12m"] = feats["sum_3m"] / (feats["sum_12m"] + eps)
    feats["ratio_6m_12m"] = feats["sum_6m"] / (feats["sum_12m"] + eps)
    feats["ratio_6m_24m"] = feats["sum_6m"] / (feats["sum_24m"] + eps)
    feats["ratio_12m_24m"] = feats["sum_12m"] / (feats["sum_24m"] + eps)
    feats["ratio_recent_to_lifetime"] = (
        _safe_window(history, 3).mean() / (history.mean() + eps)
    )
    feats["mean_ratio_3m_12m"] = feats["mean_3m"] / (feats["mean_12m"] + eps)
    feats["mean_ratio_6m_12m"] = feats["mean_6m"] / (feats["mean_12m"] + eps)
    feats["mean_ratio_1m_6m"] = feats["mean_1m"] / (feats["mean_6m"] + eps)

    # ==================================================================
    # CATEGORY 4: YEAR-OVER-YEAR
    # ==================================================================
    if tenure >= 24:
        prev_12m = history[-24:-12].sum()
        feats["yoy_growth"] = (feats["sum_12m"] - prev_12m) / (prev_12m + eps)
        feats["prev_12m_sum"] = prev_12m
        feats["prev_12m_mean"] = history[-24:-12].mean()
        feats["yoy_mean_growth"] = (
            feats["mean_12m"] - feats["prev_12m_mean"]
        ) / (feats["prev_12m_mean"] + eps)
    else:
        feats["yoy_growth"] = 0.0
        feats["prev_12m_sum"] = feats["sum_12m"]
        feats["prev_12m_mean"] = feats["mean_12m"]
        feats["yoy_mean_growth"] = 0.0

    if tenure >= 36:
        pp12 = history[-36:-24].sum()
        feats["yoy_growth_2y_ago"] = (
            history[-24:-12].sum() - pp12
        ) / (pp12 + eps)
        feats["yoy_acceleration"] = feats["yoy_growth"] - feats["yoy_growth_2y_ago"]
    else:
        feats["yoy_growth_2y_ago"] = 0.0
        feats["yoy_acceleration"] = 0.0

    if tenure >= 13:
        feats["same_month_yoy"] = history[-1] / (history[-13] + eps)
    else:
        feats["same_month_yoy"] = 1.0

    # ==================================================================
    # CATEGORY 5: TREND (slope, momentum, acceleration)
    # ==================================================================
    for label, lb in [("full", min(tenure, 24)), ("12m", min(tenure, 12)),
                       ("6m", min(tenure, 6)), ("3m", min(tenure, 3))]:
        seg = history[-lb:]
        x = np.arange(lb, dtype=float)
        if lb > 2:
            coeffs = np.polyfit(x, seg, 1)
            feats[f"slope_{label}"] = coeffs[0]
            feats[f"slope_{label}_norm"] = coeffs[0] / (seg.mean() + eps)
            predicted = np.polyval(coeffs, x)
            ss_res = np.sum((seg - predicted) ** 2)
            ss_tot = np.sum((seg - seg.mean()) ** 2)
            feats[f"r2_{label}"] = 1 - ss_res / (ss_tot + eps)
        else:
            feats[f"slope_{label}"] = 0.0
            feats[f"slope_{label}_norm"] = 0.0
            feats[f"r2_{label}"] = 0.0

    if tenure >= 12:
        feats["momentum_6m"] = (
            history[-6:].mean() - history[-12:-6].mean()
        ) / (history[-12:-6].mean() + eps)
    else:
        feats["momentum_6m"] = 0.0

    if tenure >= 24:
        feats["momentum_12m"] = (
            history[-12:].mean() - history[-24:-12].mean()
        ) / (history[-24:-12].mean() + eps)
    else:
        feats["momentum_12m"] = 0.0

    if tenure >= 12:
        h1 = history[-12:-6]
        h2 = history[-6:]
        s1 = np.polyfit(np.arange(len(h1), dtype=float), h1, 1)[0] if len(h1) > 2 else 0.0
        s2 = np.polyfit(np.arange(len(h2), dtype=float), h2, 1)[0] if len(h2) > 2 else 0.0
        feats["slope_acceleration"] = s2 - s1
        feats["slope_acceleration_norm"] = (s2 - s1) / (history[-12:].mean() + eps)
    else:
        feats["slope_acceleration"] = 0.0
        feats["slope_acceleration_norm"] = 0.0

    # ==================================================================
    # CATEGORY 6: DIFF / CHANGE FEATURES
    # ==================================================================
    if tenure >= 3:
        diffs = np.diff(history[-min(13, tenure):])
        abs_diffs = np.abs(diffs)
        feats["mean_diff"] = diffs.mean()
        feats["std_diff"] = diffs.std()
        feats["last_diff"] = diffs[-1]
        feats["mean_abs_diff"] = abs_diffs.mean()
        feats["max_abs_diff"] = abs_diffs.max()
        feats["median_abs_diff"] = np.median(abs_diffs)
        shifted = history[-min(13, tenure):-1]
        pct_changes = diffs / (np.abs(shifted) + eps)
        feats["mean_pct_change"] = pct_changes.mean()
        feats["std_pct_change"] = pct_changes.std()
        feats["max_pct_change"] = pct_changes.max()
        feats["min_pct_change"] = pct_changes.min()
        feats["last_pct_change"] = pct_changes[-1]
        feats["median_pct_change"] = np.median(pct_changes)
        feats["pct_months_positive"] = (diffs > 0).mean()
        feats["pct_months_negative"] = (diffs < 0).mean()
        feats["pct_months_flat"] = (diffs == 0).mean()
    else:
        for f in ["mean_diff", "std_diff", "last_diff", "mean_abs_diff",
                   "max_abs_diff", "median_abs_diff", "mean_pct_change",
                   "std_pct_change", "max_pct_change", "min_pct_change",
                   "last_pct_change", "median_pct_change",
                   "pct_months_positive", "pct_months_negative", "pct_months_flat"]:
            feats[f] = 0.0

    # ==================================================================
    # CATEGORY 7: SEASONALITY
    # ==================================================================
    origin_month = origin_date.month
    feats["origin_month_sin"] = np.sin(2 * np.pi * origin_month / 12)
    feats["origin_month_cos"] = np.cos(2 * np.pi * origin_month / 12)
    feats["origin_month"] = origin_month
    feats["origin_quarter"] = (origin_month - 1) // 3

    if tenure >= 24:
        last_24 = history[-24:]
        monthly_means = np.array([last_24[i::12].mean() for i in range(12)])
        overall_mean = monthly_means.mean()
        seasonal_idx = monthly_means / (overall_mean + eps)

        feats["seasonal_strength"] = monthly_means.std() / (overall_mean + eps)
        feats["seasonal_range"] = (monthly_means.max() - monthly_means.min()) / (overall_mean + eps)

        origin_pos = origin_month - 1
        forecast_positions = [(origin_pos + i) % 12 for i in range(1, 13)]
        feats["forecast_seasonal_load"] = seasonal_idx[forecast_positions].mean()
        feats["forecast_seasonal_std"] = seasonal_idx[forecast_positions].std()
        feats["forecast_h1_seasonal"] = seasonal_idx[forecast_positions[:6]].mean()
        feats["forecast_h2_seasonal"] = seasonal_idx[forecast_positions[6:]].mean()

        peak_pos = int(np.argmax(monthly_means))
        trough_pos = int(np.argmin(monthly_means))
        feats["months_to_peak"] = (peak_pos - origin_pos) % 12
        feats["months_to_trough"] = (trough_pos - origin_pos) % 12
        feats["peak_trough_ratio"] = monthly_means[peak_pos] / (monthly_means[trough_pos] + eps)

        yr1 = history[-24:-12]
        yr2 = history[-12:]
        yr1_norm = yr1 / (yr1.mean() + eps)
        yr2_norm = yr2 / (yr2.mean() + eps)
        corr = np.corrcoef(yr1_norm, yr2_norm)[0, 1]
        feats["seasonal_consistency"] = corr if not np.isnan(corr) else 0.0
    elif tenure >= 13:
        feats["same_month_yoy_ratio"] = history[-1] / (history[-13] + eps)
        for f in ["seasonal_strength", "seasonal_range", "forecast_seasonal_std",
                   "seasonal_consistency"]:
            feats[f] = 0.0
        for f in ["forecast_seasonal_load", "forecast_h1_seasonal", "forecast_h2_seasonal"]:
            feats[f] = 1.0
        feats["months_to_peak"] = 0
        feats["months_to_trough"] = 0
        feats["peak_trough_ratio"] = 1.0
    else:
        for f in ["seasonal_strength", "seasonal_range", "forecast_seasonal_std",
                   "seasonal_consistency"]:
            feats[f] = 0.0
        for f in ["forecast_seasonal_load", "forecast_h1_seasonal", "forecast_h2_seasonal"]:
            feats[f] = 1.0
        feats["months_to_peak"] = 0
        feats["months_to_trough"] = 0
        feats["peak_trough_ratio"] = 1.0

    # ==================================================================
    # CATEGORY 8: VOLATILITY / DISTRIBUTION SHAPE
    # ==================================================================
    for label, w in [("12m", 12), ("6m", 6), ("lifetime", tenure)]:
        win = _safe_window(history, w)
        m = win.mean()
        feats[f"cv_{label}"] = win.std() / (m + eps)
        feats[f"iqr_{label}"] = np.percentile(win, 75) - np.percentile(win, 25)
        feats[f"iqr_pct_{label}"] = feats[f"iqr_{label}"] / (m + eps)
        if len(win) >= 4:
            feats[f"skew_{label}"] = float(pd.Series(win).skew())
            feats[f"kurtosis_{label}"] = float(pd.Series(win).kurtosis())
        else:
            feats[f"skew_{label}"] = 0.0
            feats[f"kurtosis_{label}"] = 0.0

    r12 = _safe_window(history, 12)
    feats["mad_12m"] = np.mean(np.abs(r12 - r12.mean()))
    feats["mad_12m_pct"] = feats["mad_12m"] / (r12.mean() + eps)

    # ==================================================================
    # CATEGORY 9: OUTLIER DETECTION & CHARACTERIZATION
    # ==================================================================
    for label, w in [("12m", 12), ("24m", 24)]:
        win = _safe_window(history, w)
        if len(win) >= 4:
            q1, q3 = np.percentile(win, 25), np.percentile(win, 75)
            iqr = q3 - q1
            upper_fence = q3 + 1.5 * iqr
            lower_fence = q1 - 1.5 * iqr

            high_outliers = win[win > upper_fence]
            low_outliers = win[win < lower_fence]

            feats[f"outlier_high_count_{label}"] = len(high_outliers)
            feats[f"outlier_low_count_{label}"] = len(low_outliers)
            feats[f"outlier_total_count_{label}"] = len(high_outliers) + len(low_outliers)
            feats[f"outlier_frac_{label}"] = feats[f"outlier_total_count_{label}"] / len(win)

            if len(high_outliers) > 0:
                feats[f"outlier_high_max_{label}"] = high_outliers.max()
                feats[f"outlier_high_max_pct_{label}"] = high_outliers.max() / (win.mean() + eps)
                feats[f"outlier_high_mean_pct_{label}"] = high_outliers.mean() / (win.mean() + eps)
            else:
                feats[f"outlier_high_max_{label}"] = 0.0
                feats[f"outlier_high_max_pct_{label}"] = 0.0
                feats[f"outlier_high_mean_pct_{label}"] = 0.0

            if len(low_outliers) > 0:
                feats[f"outlier_low_min_{label}"] = low_outliers.min()
                feats[f"outlier_low_min_pct_{label}"] = low_outliers.min() / (win.mean() + eps)
            else:
                feats[f"outlier_low_min_{label}"] = 0.0
                feats[f"outlier_low_min_pct_{label}"] = 0.0

            # Clustered (regime change) vs isolated (one-time)?
            all_outlier_mask = (win > upper_fence) | (win < lower_fence)
            if all_outlier_mask.sum() >= 2:
                outlier_positions = np.where(all_outlier_mask)[0]
                gaps = np.diff(outlier_positions)
                feats[f"outlier_clustered_{label}"] = int(np.any(gaps == 1))
                feats[f"outlier_mean_gap_{label}"] = gaps.mean()
            else:
                feats[f"outlier_clustered_{label}"] = 0
                feats[f"outlier_mean_gap_{label}"] = float(len(win))

            feats[f"last_month_is_outlier_{label}"] = int(
                win[-1] > upper_fence or win[-1] < lower_fence
            )
            last3 = win[-min(3, len(win)):]
            feats[f"recent3_outlier_count_{label}"] = int(
                np.sum((last3 > upper_fence) | (last3 < lower_fence))
            )
        else:
            for f in [f"outlier_high_count_{label}", f"outlier_low_count_{label}",
                       f"outlier_total_count_{label}", f"outlier_frac_{label}",
                       f"outlier_high_max_{label}", f"outlier_high_max_pct_{label}",
                       f"outlier_high_mean_pct_{label}", f"outlier_low_min_{label}",
                       f"outlier_low_min_pct_{label}", f"outlier_clustered_{label}",
                       f"outlier_mean_gap_{label}", f"last_month_is_outlier_{label}",
                       f"recent3_outlier_count_{label}"]:
                feats[f] = 0.0

    # Z-score based (robust: using median/MAD)
    r12 = _safe_window(history, 12)
    med = np.median(r12)
    mad = np.median(np.abs(r12 - med)) + eps
    z_scores = (r12 - med) / (1.4826 * mad)
    feats["zscore_last_month"] = z_scores[-1]
    feats["zscore_max_12m"] = z_scores.max()
    feats["zscore_min_12m"] = z_scores.min()
    feats["zscore_abs_max_12m"] = np.abs(z_scores).max()
    feats["months_with_zscore_gt2"] = int(np.sum(np.abs(z_scores) > 2))
    feats["months_with_zscore_gt3"] = int(np.sum(np.abs(z_scores) > 3))

    # Winsorized mean vs regular mean
    if len(r12) >= 4:
        sorted_vals = np.sort(r12)
        trim = max(1, len(sorted_vals) // 10)
        winsorized_mean = sorted_vals[trim:-trim].mean()
        feats["winsorized_mean_12m"] = winsorized_mean
        feats["winsorized_vs_mean_ratio"] = winsorized_mean / (r12.mean() + eps)
    else:
        feats["winsorized_mean_12m"] = r12.mean()
        feats["winsorized_vs_mean_ratio"] = 1.0

    # ==================================================================
    # CATEGORY 10: SPARSITY / ZERO PATTERNS
    # ==================================================================
    for label, w in [("3m", 3), ("6m", 6), ("12m", 12), ("24m", 24), ("life", tenure)]:
        win = _safe_window(history, w)
        n_zeros = int((win == 0).sum())
        feats[f"zero_count_{label}"] = n_zeros
        feats[f"zero_frac_{label}"] = n_zeros / len(win)
        threshold = win.mean() * 0.01
        feats[f"near_zero_count_{label}"] = int((win < threshold).sum())
        feats[f"near_zero_frac_{label}"] = (win < threshold).mean()

    # Consecutive zero runs analysis
    is_zero = (history == 0).astype(int)
    all_runs = []
    current_run = 0
    for val in is_zero:
        if val:
            current_run += 1
        else:
            if current_run > 0:
                all_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        all_runs.append(current_run)

    feats["max_consec_zeros"] = max(all_runs) if all_runs else 0
    feats["num_zero_runs"] = len(all_runs)
    feats["mean_zero_run_length"] = np.mean(all_runs) if all_runs else 0.0
    feats["median_zero_run_length"] = np.median(all_runs) if all_runs else 0.0

    # Trailing zeros
    feats["ending_on_zeros"] = int(history[-1] == 0)
    trailing_zeros = 0
    for v in reversed(history):
        if v == 0:
            trailing_zeros += 1
        else:
            break
    feats["trailing_zero_count"] = trailing_zeros
    feats["trailing_zero_frac"] = trailing_zeros / tenure

    # Months since last zero / nonzero
    zero_positions = np.where(history == 0)[0]
    feats["months_since_last_zero"] = (tenure - 1 - zero_positions[-1]) if len(zero_positions) > 0 else tenure
    nonzero_positions = np.where(history != 0)[0]
    feats["months_since_last_nonzero"] = (tenure - 1 - nonzero_positions[-1]) if len(nonzero_positions) > 0 else tenure

    # Flags
    feats["is_sparse"] = int(feats["zero_frac_12m"] > 0.25)
    feats["is_intermittent"] = int(
        feats["zero_count_12m"] > 0 and feats["trailing_zero_count"] < feats["zero_count_12m"]
    )

    # Nonzero-only stats
    r12 = _safe_window(history, 12)
    nonzero_vals = r12[r12 > 0]
    if len(nonzero_vals) > 0:
        feats["mean_nonzero_12m"] = nonzero_vals.mean()
        feats["std_nonzero_12m"] = nonzero_vals.std()
        feats["cv_nonzero_12m"] = nonzero_vals.std() / (nonzero_vals.mean() + eps)
        feats["sum_nonzero_12m"] = nonzero_vals.sum()
        feats["count_nonzero_12m"] = len(nonzero_vals)
    else:
        feats["mean_nonzero_12m"] = 0.0
        feats["std_nonzero_12m"] = 0.0
        feats["cv_nonzero_12m"] = 0.0
        feats["sum_nonzero_12m"] = 0.0
        feats["count_nonzero_12m"] = 0

    # ==================================================================
    # CATEGORY 11: STRUCTURAL BREAK / REGIME CHANGE
    # ==================================================================
    if tenure >= 12:
        r12 = history[-12:]
        cumdev = np.cumsum(r12 - r12.mean())
        feats["cusum_range_12m"] = cumdev.max() - cumdev.min()
        feats["cusum_range_pct_12m"] = feats["cusum_range_12m"] / (r12.sum() + eps)

        best_split_diff = 0.0
        best_split_pos = 0
        for sp in range(3, len(r12) - 2):
            left_mean = r12[:sp].mean()
            right_mean = r12[sp:].mean()
            diff = abs(right_mean - left_mean) / (r12.mean() + eps)
            if diff > best_split_diff:
                best_split_diff = diff
                best_split_pos = sp
        feats["regime_change_magnitude_12m"] = best_split_diff
        feats["regime_change_position_12m"] = best_split_pos
        feats["regime_change_recency_12m"] = 12 - best_split_pos

        if 0 < best_split_pos < 12:
            pre = r12[:best_split_pos].mean()
            post = r12[best_split_pos:].mean()
            feats["regime_change_direction_12m"] = (post - pre) / (pre + eps)
        else:
            feats["regime_change_direction_12m"] = 0.0
    else:
        feats["cusum_range_12m"] = 0.0
        feats["cusum_range_pct_12m"] = 0.0
        feats["regime_change_magnitude_12m"] = 0.0
        feats["regime_change_position_12m"] = 0
        feats["regime_change_recency_12m"] = 0
        feats["regime_change_direction_12m"] = 0.0

    if tenure >= 24:
        first_12 = history[-24:-12].mean()
        second_12 = history[-12:].mean()
        feats["regime_shift_24m"] = (second_12 - first_12) / (first_12 + eps)
        feats["regime_shift_24m_abs"] = abs(second_12 - first_12) / (first_12 + eps)
    else:
        feats["regime_shift_24m"] = 0.0
        feats["regime_shift_24m_abs"] = 0.0

    # ==================================================================
    # CATEGORY 12: RECENCY & LEVEL-SHIFT
    # ==================================================================
    last_val = history[-1]
    feats["last_val"] = last_val
    feats["last_vs_mean_3m"] = last_val / (feats["mean_3m"] + eps)
    feats["last_vs_mean_6m"] = last_val / (feats["mean_6m"] + eps)
    feats["last_vs_mean_12m"] = last_val / (feats["mean_12m"] + eps)
    feats["last_vs_median_12m"] = last_val / (feats["median_12m"] + eps)
    feats["last_vs_lifetime_mean"] = last_val / (feats["lifetime_mean"] + eps)

    last2 = _safe_window(history, 2).mean()
    feats["last2_vs_mean_12m"] = last2 / (feats["mean_12m"] + eps)
    feats["last2_vs_mean_6m"] = last2 / (feats["mean_6m"] + eps)

    last3_mean = _safe_window(history, 3).mean()
    feats["last3_vs_mean_12m"] = last3_mean / (feats["mean_12m"] + eps)
    feats["last3_vs_mean_6m"] = last3_mean / (feats["mean_6m"] + eps)

    if tenure >= 2:
        prev_val = history[-2]
        feats["last_vs_prev_month"] = last_val / (prev_val + eps)
        feats["last_minus_prev_abs"] = abs(last_val - prev_val)
        feats["last_minus_prev_pct"] = (last_val - prev_val) / (prev_val + eps)
    else:
        feats["last_vs_prev_month"] = 1.0
        feats["last_minus_prev_abs"] = 0.0
        feats["last_minus_prev_pct"] = 0.0

    # ==================================================================
    # CATEGORY 13: PERCENTILE / RANK FEATURES
    # ==================================================================
    r12 = _safe_window(history, 12)
    feats["last_percentile_12m"] = np.mean(r12 <= last_val)
    feats["last3_percentile_12m"] = np.mean(r12 <= last3_mean)

    for p in [10, 25, 50, 75, 90]:
        feats[f"pctl_{p}_12m"] = np.percentile(r12, p)
        feats[f"pctl_{p}_pct_mean_12m"] = feats[f"pctl_{p}_12m"] / (r12.mean() + eps)

    # ==================================================================
    # CATEGORY 14: CONCENTRATION / GINI
    # ==================================================================
    r12 = _safe_window(history, 12)
    sorted_vals = np.sort(np.abs(r12))
    n = len(sorted_vals)
    cumvals = np.cumsum(sorted_vals)
    total = sorted_vals.sum()
    if total > 0 and n > 1:
        feats["gini_12m"] = (2 * np.sum(cumvals) - total * (n + 1)) / (n * total)
    else:
        feats["gini_12m"] = 0.0

    sorted_desc = np.sort(r12)[::-1]
    total_12m = r12.sum() + eps
    feats["top1_month_share"] = sorted_desc[0] / total_12m
    feats["top3_month_share"] = sorted_desc[:min(3, n)].sum() / total_12m
    feats["bottom3_month_share"] = sorted_desc[-min(3, n):].sum() / total_12m

    # ==================================================================
    # CATEGORY 15: LOG-SCALE FEATURES
    # ==================================================================
    feats["log_mean_12m"] = np.log1p(max(feats["mean_12m"], 0))
    feats["log_sum_12m"] = np.log1p(max(feats["sum_12m"], 0))
    feats["log_mean_6m"] = np.log1p(max(feats["mean_6m"], 0))
    feats["log_lifetime_mean"] = np.log1p(max(feats["lifetime_mean"], 0))
    feats["log_last_val"] = np.log1p(max(last_val, 0))
    feats["log_annualized_rate"] = np.log1p(max(feats["annualized_rate"], 0))
    feats["log_std_12m"] = np.log1p(max(feats["std_12m"], 0))

    # ==================================================================
    # CATEGORY 16: AUTOCORRELATION FEATURES
    # ==================================================================
    if tenure >= 6:
        r = _safe_window(history, min(tenure, 24))
        mean_r = r.mean()
        var_r = np.var(r) + eps
        for lag in [1, 2, 3, 6, 12]:
            if len(r) > lag:
                ac = np.mean((r[lag:] - mean_r) * (r[:-lag] - mean_r)) / var_r
                feats[f"autocorr_lag{lag}"] = ac
            else:
                feats[f"autocorr_lag{lag}"] = 0.0
    else:
        for lag in [1, 2, 3, 6, 12]:
            feats[f"autocorr_lag{lag}"] = 0.0

    # ==================================================================
    # CATEGORY 17: ROLLING VOLATILITY CHANGE
    # ==================================================================
    if tenure >= 12:
        first_half_std = history[-12:-6].std()
        second_half_std = history[-6:].std()
        feats["vol_change_6m"] = (second_half_std - first_half_std) / (first_half_std + eps)
        feats["vol_ratio_6m"] = second_half_std / (first_half_std + eps)
    else:
        feats["vol_change_6m"] = 0.0
        feats["vol_ratio_6m"] = 1.0

    if tenure >= 24:
        feats["vol_change_12m"] = (
            history[-12:].std() - history[-24:-12].std()
        ) / (history[-24:-12].std() + eps)
    else:
        feats["vol_change_12m"] = 0.0

    # ==================================================================
    # CATEGORY 18: EXTREME VALUE FEATURES
    # ==================================================================
    r12 = _safe_window(history, 12)
    feats["max_vs_mean_12m"] = r12.max() / (r12.mean() + eps)
    feats["min_vs_mean_12m"] = r12.min() / (r12.mean() + eps)
    feats["max_vs_median_12m"] = r12.max() / (np.median(r12) + eps)
    feats["range_12m"] = r12.max() - r12.min()
    feats["range_pct_12m"] = feats["range_12m"] / (r12.mean() + eps)
    feats["months_since_max_12m"] = len(r12) - 1 - np.argmax(r12)
    feats["months_since_min_12m"] = len(r12) - 1 - np.argmin(r12)
    feats["months_since_lifetime_max"] = tenure - 1 - np.argmax(history)
    feats["months_since_lifetime_min"] = tenure - 1 - np.argmin(history)
    feats["lifetime_max"] = history.max()
    feats["lifetime_min"] = history.min()
    feats["lifetime_max_vs_mean"] = history.max() / (history.mean() + eps)

    return feats


# =============================================================================
# 3. TRAINING DATASET CONSTRUCTION
# =============================================================================

def build_training_data(
    df: pd.DataFrame,
    min_history_months: int = 6,
    forecast_horizon: int = 12,
    max_origins_per_tin: Optional[int] = None,
) -> pd.DataFrame:
    """Build training dataset using expanding window approach."""
    all_rows = []
    for tin, grp in df.groupby("tin"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        cashflows = grp["y"].values
        dates = grp["ds"].values
        n = len(cashflows)

        first_origin = min_history_months - 1
        last_origin = n - forecast_horizon - 1
        if first_origin > last_origin:
            continue

        origins = list(range(first_origin, last_origin + 1))
        if max_origins_per_tin and len(origins) > max_origins_per_tin:
            origins = origins[-max_origins_per_tin:]

        for oidx in origins:
            target = cashflows[oidx + 1: oidx + 1 + forecast_horizon].sum()
            origin_date = pd.Timestamp(dates[oidx])
            feats = make_features_for_tin(cashflows, oidx, origin_date)
            feats["tin"] = tin
            feats["origin_date"] = origin_date
            feats["target"] = target
            all_rows.append(feats)

    return pd.DataFrame(all_rows)


def build_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tin using all available history."""
    all_rows = []
    for tin, grp in df.groupby("tin"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        cashflows = grp["y"].values
        dates = grp["ds"].values
        origin_idx = len(cashflows) - 1
        origin_date = pd.Timestamp(dates[origin_idx])
        feats = make_features_for_tin(cashflows, origin_idx, origin_date)
        feats["tin"] = tin
        feats["origin_date"] = origin_date
        all_rows.append(feats)
    return pd.DataFrame(all_rows)


# =============================================================================
# 4. TRAIN / VALIDATION SPLIT
# =============================================================================

def temporal_train_val_split(train_df: pd.DataFrame, val_months: int = 12) -> tuple:
    cutoff = train_df["origin_date"].max() - pd.DateOffset(months=val_months)
    train_set = train_df[train_df["origin_date"] <= cutoff].copy()
    val_set = train_df[train_df["origin_date"] > cutoff].copy()
    print(f"Train: {len(train_set)} rows, origins up to {cutoff.date()}")
    print(f"Val:   {len(val_set)} rows, origins after {cutoff.date()}")
    return train_set, val_set


# =============================================================================
# 5. FEATURE SELECTION
# =============================================================================

NON_FEATURE_COLS = {"tin", "origin_date", "target"}


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def select_features(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    importance_type: str = "gain",
    corr_threshold: float = 0.95,
    min_importance_pct: float = 0.001,
    top_n: Optional[int] = None,
    lgb_params: Optional[dict] = None,
    categorical_features: Optional[list] = None,
) -> tuple:
    """
    Multi-stage feature selection:
      Stage 1: Remove near-zero-variance features
      Stage 2: Remove highly correlated features (keep the more important one)
      Stage 3: Train a preliminary LightGBM, drop features with negligible importance
      Stage 4: Optionally cap at top_n features

    Parameters
    ----------
    train_set / val_set : DataFrames with features + target
    importance_type : 'gain' or 'split' for LightGBM importance
    corr_threshold : drop one of any pair with |corr| above this
    min_importance_pct : drop features contributing less than this % of total importance
    top_n : if set, keep only the top N features (None = keep all that pass filters)
    lgb_params : LightGBM params for the preliminary model
    categorical_features : list of categorical feature names

    Returns
    -------
    (selected_features: list, selection_report: pd.DataFrame)
    """
    all_features = get_feature_cols(train_set)
    print(f"\n  Starting features: {len(all_features)}")

    # ── Stage 1: Remove near-zero-variance ──
    # Features where >99% of values are identical
    drop_nzv = []
    for col in all_features:
        if train_set[col].dtype in ("object", "str"):
            continue
        top_frac = train_set[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_frac > 0.99:
            drop_nzv.append(col)
    remaining = [f for f in all_features if f not in drop_nzv]
    print(f"  After near-zero-variance removal: {len(remaining)} "
          f"(dropped {len(drop_nzv)})")

    # ── Stage 2: Remove highly correlated features ──
    # Build correlation matrix on numeric features only
    numeric_remaining = [c for c in remaining
                         if train_set[c].dtype not in ("object", "str")]
    corr_matrix = train_set[numeric_remaining].corr().abs()

    # Train a quick model to get importance for tie-breaking
    if lgb_params is None:
        quick_params = {
            "objective": "regression", "metric": "mae",
            "learning_rate": 0.1, "num_leaves": 63,
            "min_child_samples": 50, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "verbosity": -1, "n_jobs": -1, "seed": 42,
        }
    else:
        quick_params = lgb_params.copy()
        quick_params["learning_rate"] = 0.1

    if categorical_features is None:
        categorical_features = ["origin_month", "origin_quarter", "tenure_bucket"]
    cat_feats = [c for c in categorical_features if c in remaining]

    dtrain = lgb.Dataset(
        train_set[remaining], label=train_set["target"],
        categorical_feature=cat_feats, free_raw_data=False,
    )
    dval = lgb.Dataset(
        val_set[remaining], label=val_set["target"],
        categorical_feature=cat_feats, reference=dtrain, free_raw_data=False,
    )
    preliminary_model = lgb.train(
        quick_params, dtrain, num_boost_round=500,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    # Build importance lookup
    imp_values = preliminary_model.feature_importance(importance_type=importance_type)
    imp_lookup = dict(zip(preliminary_model.feature_name(), imp_values))

    # Greedy corr removal: for each highly correlated pair, drop the less important one
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )
    drop_corr = set()
    for col in upper_tri.columns:
        for row in upper_tri.index:
            if upper_tri.loc[row, col] > corr_threshold:
                imp_row = imp_lookup.get(row, 0)
                imp_col = imp_lookup.get(col, 0)
                drop_feat = row if imp_row < imp_col else col
                drop_corr.add(drop_feat)

    remaining = [f for f in remaining if f not in drop_corr]
    print(f"  After correlation removal (>{corr_threshold}): {len(remaining)} "
          f"(dropped {len(drop_corr)})")

    # ── Stage 3: Remove low-importance features ──
    total_imp = sum(imp_lookup.get(f, 0) for f in remaining)
    if total_imp > 0:
        drop_low_imp = []
        for f in remaining:
            feat_imp_pct = imp_lookup.get(f, 0) / total_imp
            if feat_imp_pct < min_importance_pct:
                drop_low_imp.append(f)
        remaining = [f for f in remaining if f not in drop_low_imp]
        print(f"  After low-importance removal (<{min_importance_pct*100}% of total): "
              f"{len(remaining)} (dropped {len(drop_low_imp)})")

    # ── Stage 4: Optional top-N cap ──
    if top_n and len(remaining) > top_n:
        ranked = sorted(remaining, key=lambda f: imp_lookup.get(f, 0), reverse=True)
        remaining = ranked[:top_n]
        print(f"  After top-N cap: {len(remaining)}")

    # ── Build selection report ──
    report_rows = []
    for f in all_features:
        imp = imp_lookup.get(f, 0)
        imp_pct = imp / total_imp * 100 if total_imp > 0 else 0
        if f in drop_nzv:
            status = "dropped_near_zero_variance"
        elif f in drop_corr:
            status = "dropped_high_correlation"
        elif f in drop_low_imp if total_imp > 0 else False:
            status = "dropped_low_importance"
        elif f not in remaining and top_n:
            status = "dropped_top_n_cap"
        elif f in remaining:
            status = "selected"
        else:
            status = "dropped"
        report_rows.append({
            "feature": f,
            "importance": imp,
            "importance_pct": round(imp_pct, 4),
            "status": status,
        })
    report = (
        pd.DataFrame(report_rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    n_selected = (report["status"] == "selected").sum()
    print(f"\n  FINAL: {n_selected} features selected out of {len(all_features)}")

    return remaining, report


# =============================================================================
# 6. TEMPORAL CROSS-VALIDATION & HYPERPARAMETER TUNING
# =============================================================================


def temporal_cv(
    train_data: pd.DataFrame,
    feature_cols: list,
    lgb_params: dict,
    n_folds: int = 3,
    fold_gap_months: int = 0,
    categorical_features: Optional[list] = None,
) -> dict:
    """
    Expanding-window temporal cross-validation.

    Instead of random splits, each fold uses an earlier cutoff for training
    and the next chunk for validation, preserving time ordering:
        Fold 1: train on [oldest ... T1],  validate on [T1 ... T2]
        Fold 2: train on [oldest ... T2],  validate on [T2 ... T3]
        Fold 3: train on [oldest ... T3],  validate on [T3 ... latest]

    Parameters
    ----------
    train_data : full training DataFrame (with target)
    feature_cols : list of features to use
    lgb_params : LightGBM parameters
    n_folds : number of temporal folds (default 3)
    fold_gap_months : gap between train and val to prevent leakage (default 0)
    categorical_features : categorical feature names

    Returns
    -------
    dict with keys: fold_metrics (list of dicts), mean_mae, std_mae, mean_best_iter
    """
    if categorical_features is None:
        categorical_features = ["origin_month", "origin_quarter", "tenure_bucket"]
    cat_feats = [c for c in categorical_features if c in feature_cols]

    # Create temporal fold boundaries
    all_dates = sorted(train_data["origin_date"].unique())
    n_dates = len(all_dates)
    # Reserve the last (n_folds) chunks as validation folds
    # Each fold gets roughly equal number of validation dates
    val_size = n_dates // (n_folds + 1)  # +1 so first chunk is train-only

    fold_metrics = []
    for fold_idx in range(n_folds):
        # Validation dates for this fold
        val_start_idx = n_dates - (n_folds - fold_idx) * val_size
        val_end_idx = n_dates - (n_folds - fold_idx - 1) * val_size if fold_idx < n_folds - 1 else n_dates
        val_start_date = all_dates[val_start_idx]

        if fold_gap_months > 0:
            train_end_date = val_start_date - pd.DateOffset(months=fold_gap_months)
        else:
            train_end_date = val_start_date

        fold_train = train_data[train_data["origin_date"] < train_end_date]
        fold_val = train_data[
            (train_data["origin_date"] >= val_start_date) &
            (train_data["origin_date"] < all_dates[val_end_idx] if val_end_idx < n_dates else True)
        ]

        if len(fold_train) < 50 or len(fold_val) < 10:
            continue

        dtrain = lgb.Dataset(
            fold_train[feature_cols], label=fold_train["target"],
            categorical_feature=cat_feats, free_raw_data=False,
        )
        dval = lgb.Dataset(
            fold_val[feature_cols], label=fold_val["target"],
            categorical_feature=cat_feats, reference=dtrain, free_raw_data=False,
        )

        model = lgb.train(
            lgb_params, dtrain, num_boost_round=3000,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(fold_val[feature_cols], num_iteration=model.best_iteration)
        mae = np.mean(np.abs(fold_val["target"].values - preds))
        ape = np.abs(fold_val["target"].values - preds) / (np.abs(fold_val["target"].values) + 1e-9)
        median_ape = np.median(ape)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "train_rows": len(fold_train),
            "val_rows": len(fold_val),
            "best_iteration": model.best_iteration,
            "mae": mae,
            "median_ape": median_ape,
        })

    mean_mae = np.mean([f["mae"] for f in fold_metrics])
    std_mae = np.std([f["mae"] for f in fold_metrics])
    mean_ape = np.mean([f["median_ape"] for f in fold_metrics])
    mean_iter = np.mean([f["best_iteration"] for f in fold_metrics])

    return {
        "fold_metrics": fold_metrics,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "mean_median_ape": mean_ape,
        "mean_best_iter": int(mean_iter),
    }


def tune_hyperparameters(
    train_data: pd.DataFrame,
    feature_cols: list,
    n_folds: int = 3,
    n_trials: int = 50,
    categorical_features: Optional[list] = None,
) -> tuple:
    """
    Random search over LightGBM hyperparameters using temporal CV.

    Searches over: min_child_samples, feature_fraction, bagging_fraction,
    reg_alpha, reg_lambda, learning_rate, max_depth (capped at 3-7).
    num_leaves is derived from max_depth (2^depth - 1), not tuned independently.

    Parameters
    ----------
    train_data : full training DataFrame
    feature_cols : features to use
    n_folds : temporal CV folds
    n_trials : number of random parameter combos to try
    categorical_features : categorical feature names

    Returns
    -------
    (best_params, tuning_log)
        best_params: dict ready to pass to train_lightgbm
        tuning_log: DataFrame with all trials
    """
    rng = np.random.RandomState(42)

    # Define search space
    # num_leaves is fixed (not tuned) — controlled by max_depth instead.
    # max_depth is capped below 8 to prevent overfitting.
    def _sample_params():
        max_depth = int(rng.choice([3, 4, 5, 6, 7, 8]))
        return {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "n_jobs": -1,
            "seed": rng.randint(0, 10000),
            "learning_rate": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.1]),
            "num_leaves": 2 ** max_depth - 1,  # derived from max_depth, not tuned independently
            "max_depth": max_depth,
            "min_child_samples": int(rng.choice([10, 20, 30, 50, 100])),
            "feature_fraction": rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "bagging_fraction": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
            "bagging_freq": int(rng.choice([0, 3, 5, 7])),
            "reg_alpha": rng.choice([0, 0.01, 0.05, 0.1, 0.5, 1.0]),
            "reg_lambda": rng.choice([0, 0.01, 0.1, 0.5, 1.0, 5.0]),
        }

    trials = []
    best_score = float("inf")
    best_params = None

    print(f"  Running {n_trials} trials with {n_folds}-fold temporal CV...")

    for trial_idx in range(n_trials):
        params = _sample_params()
        cv_result = temporal_cv(
            train_data, feature_cols, params,
            n_folds=n_folds,
            categorical_features=categorical_features,
        )

        score = cv_result["mean_mae"]
        trials.append({
            "trial": trial_idx + 1,
            "mean_mae": cv_result["mean_mae"],
            "std_mae": cv_result["std_mae"],
            "mean_median_ape": cv_result["mean_median_ape"],
            "mean_best_iter": cv_result["mean_best_iter"],
            **{k: v for k, v in params.items() if k not in ("objective", "metric", "verbosity", "n_jobs")},
        })

        if score < best_score:
            best_score = score
            best_params = params.copy()
            print(f"    Trial {trial_idx+1:3d}/{n_trials}: MAE={score:,.0f}  "
                  f"APE={cv_result['mean_median_ape']:.4f}  *** new best ***")
        elif (trial_idx + 1) % 10 == 0:
            print(f"    Trial {trial_idx+1:3d}/{n_trials}: MAE={score:,.0f}  "
                  f"(best so far: {best_score:,.0f})")

    tuning_log = pd.DataFrame(trials).sort_values("mean_mae")

    print(f"\n  Best params (MAE={best_score:,.0f}):")
    for k in ["learning_rate", "num_leaves", "max_depth", "min_child_samples",
              "feature_fraction", "bagging_fraction", "bagging_freq",
              "reg_alpha", "reg_lambda"]:
        print(f"    {k}: {best_params[k]}")

    return best_params, tuning_log


# =============================================================================
# 7. MODEL TRAINING
# =============================================================================


def train_lightgbm(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    feature_cols: Optional[list] = None,
    lgb_params: Optional[dict] = None,
    categorical_features: Optional[list] = None,
) -> lgb.Booster:
    if feature_cols is None:
        feature_cols = get_feature_cols(train_set)

    if lgb_params is None:
        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 6,
            "min_child_samples": 30,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "n_jobs": -1,
            "seed": 42,
        }

    if categorical_features is None:
        categorical_features = ["origin_month", "origin_quarter", "tenure_bucket"]
    categorical_features = [c for c in categorical_features if c in feature_cols]

    dtrain = lgb.Dataset(
        train_set[feature_cols], label=train_set["target"],
        categorical_feature=categorical_features, free_raw_data=False,
    )
    dval = lgb.Dataset(
        val_set[feature_cols], label=val_set["target"],
        categorical_feature=categorical_features, reference=dtrain, free_raw_data=False,
    )

    model = lgb.train(
        lgb_params, dtrain, num_boost_round=3000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )
    print(f"\nBest iteration: {model.best_iteration}")
    return model


# =============================================================================
# 6. PREDICTION & EVALUATION
# =============================================================================

def predict(model: lgb.Booster, data: pd.DataFrame, feature_cols: Optional[list] = None) -> np.ndarray:
    if feature_cols is None:
        feature_cols = get_feature_cols(data)
    return model.predict(data[feature_cols], num_iteration=model.best_iteration)


def compute_ape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    eps = 1e-9
    return np.abs(actual - predicted) / (np.abs(actual) + eps)


def evaluate_median_ape(
    val_set: pd.DataFrame, predictions: np.ndarray, group_col: Optional[str] = None,
) -> pd.DataFrame:
    result_df = val_set[["tin", "target"]].copy()
    result_df["predicted"] = predictions
    result_df["ape"] = compute_ape(result_df["target"].values, result_df["predicted"].values)
    overall = result_df["ape"].median()
    print(f"\nOverall Median APE: {overall:.4f} ({overall * 100:.2f}%)")

    if group_col and group_col in val_set.columns:
        result_df[group_col] = val_set[group_col].values
        group_metrics = (
            result_df.groupby(group_col)["ape"]
            .agg(["median", "mean", "count"])
            .rename(columns={"median": "median_ape", "mean": "mean_ape", "count": "n"})
            .sort_values("median_ape")
        )
        print(f"\nMedian APE by {group_col}:")
        print(group_metrics.to_string())
        return group_metrics
    return pd.DataFrame({"metric": ["median_ape"], "value": [overall]})


def evaluate_by_tin(val_set: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    result_df = val_set[["tin", "origin_date", "target"]].copy()
    result_df["predicted"] = predictions
    idx = result_df.groupby("tin")["origin_date"].idxmax()
    result_df = result_df.loc[idx].copy()
    result_df["ape"] = compute_ape(result_df["target"].values, result_df["predicted"].values)
    overall = result_df["ape"].median()
    print(f"\nTin-level Median APE (one per tin): {overall:.4f} ({overall * 100:.2f}%)")
    return result_df


# =============================================================================
# 7. FEATURE IMPORTANCE
# =============================================================================

def show_feature_importance(model: lgb.Booster, top_n: int = 50):
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    print(f"\nTop {top_n} Features (by gain):")
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:45s} {row['importance']:.1f}")
    return imp_df


# =============================================================================
# 8. FULL PIPELINE
# =============================================================================

def run_pipeline(
    df: pd.DataFrame,
    min_history_months: int = 6,
    val_months: int = 12,
    max_origins_per_tin: Optional[int] = 24,
    lgb_params: Optional[dict] = None,
    group_col: Optional[str] = None,
    # Feature selection parameters
    do_feature_selection: bool = True,
    corr_threshold: float = 0.95,
    min_importance_pct: float = 0.001,
    top_n_features: Optional[int] = None,
    # Hyperparameter tuning
    do_tune: bool = False,
    tune_n_trials: int = 50,
    tune_cv_folds: int = 3,
    # Output saving
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    End-to-end pipeline: prep -> features -> select -> tune -> train -> evaluate -> forecast.

    Parameters
    ----------
    df : raw input with columns [tin, ds, y]
    min_history_months : min history required per origin
    val_months : months of origins to hold out for validation
    max_origins_per_tin : cap training rows per tin
    lgb_params : LightGBM parameters (if None and do_tune=False, uses defaults;
                 if None and do_tune=True, uses tuned params)
    group_col : optional grouping column for evaluation
    do_feature_selection : if True, run multi-stage feature selection
    corr_threshold : drop one of any pair with |corr| above this (default 0.95)
    min_importance_pct : drop features below this % of total importance (default 0.1%)
    top_n_features : if set, cap at this many features (None = keep all that pass)
    do_tune : if True, run random search hyperparameter tuning with temporal CV
    tune_n_trials : number of random parameter combos to try (default 50)
    tune_cv_folds : number of temporal CV folds for tuning (default 3)
    output_dir : if set, save all intermediate artifacts as pickle/txt files to this dir.

    Returns
    -------
    pd.DataFrame — one row per tin with columns:
        tin, origin_date, forecast_12m_sum

    Saved files (when output_dir is set):
        01_prepared_data.pkl           - cleaned monthly df
        02_train_data_all_features.pkl - full training dataset with all features + target
        03_train_set.pkl / 04_val_set.pkl - train/val splits
        05_feature_selection_report.pkl - why each feature was kept/dropped
        06_selected_features.pkl       - list of selected feature names
        06b_tuning_log.pkl             - all tuning trials (if do_tune=True)
        06c_best_params.pkl            - best params found (if do_tune=True)
        07_lgbm_model.txt              - trained LightGBM model
        08_val_predictions.pkl         - val set with predictions and APE
        09_feature_importance.pkl      - feature importance
        10_inference_features.pkl      - inference features for all tins
        11_forecasts.pkl               - final forecast DataFrame
    """
    import pickle
    import os

    def _save(obj, filename):
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        if filename.endswith(".txt"):
            obj.save_model(filepath)
        elif filename.endswith(".pkl"):
            with open(filepath, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    -> Saved: {filepath}")

    if output_dir:
        print(f"  Artifacts will be saved to: {output_dir}/\n")

    # ── STEP 1: Prepare data ──
    print("=" * 60)
    print("STEP 1: Preparing data")
    print("=" * 60)
    df = prepare_data(df)
    df = fill_missing_months(df)
    n_tins = df["tin"].nunique()
    print(f"  {n_tins} tins, {len(df)} total monthly records")
    print(f"  Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    _save(df, "01_prepared_data.pkl")

    # ── STEP 2: Build features ──
    print("\n" + "=" * 60)
    print("STEP 2: Building training dataset")
    print("=" * 60)
    train_data = build_training_data(
        df, min_history_months=min_history_months,
        forecast_horizon=12, max_origins_per_tin=max_origins_per_tin,
    )
    n_features_raw = len(get_feature_cols(train_data))
    print(f"  {len(train_data)} training rows from {train_data['tin'].nunique()} tins")
    print(f"  {n_features_raw} raw features per row")
    _save(train_data, "02_train_data_all_features.pkl")

    # ── STEP 3: Train/Val split ──
    print("\n" + "=" * 60)
    print("STEP 3: Train/Val split")
    print("=" * 60)
    train_set, val_set = temporal_train_val_split(train_data, val_months=val_months)
    if len(val_set) == 0:
        raise ValueError("Validation set is empty.")
    _save(train_set, "03_train_set.pkl")
    _save(val_set, "04_val_set.pkl")

    # ── STEP 4: Feature selection ──
    selected_features = None
    selection_report = None
    if do_feature_selection:
        print("\n" + "=" * 60)
        print("STEP 4: Feature Selection")
        print("=" * 60)
        selected_features, selection_report = select_features(
            train_set, val_set,
            corr_threshold=corr_threshold,
            min_importance_pct=min_importance_pct,
            top_n=top_n_features,
            lgb_params=lgb_params,
        )
        _save(selection_report, "05_feature_selection_report.pkl")
        _save(selected_features, "06_selected_features.pkl")
    else:
        selected_features = get_feature_cols(train_set)
        print(f"\n  Skipping feature selection. Using all {len(selected_features)} features.")
        _save(selected_features, "06_selected_features.pkl")

    # ── STEP 5: Hyperparameter tuning ──
    tuning_log = None
    if do_tune and lgb_params is None:
        print("\n" + "=" * 60)
        print(f"STEP 5: Hyperparameter Tuning ({tune_n_trials} trials, {tune_cv_folds}-fold CV)")
        print("=" * 60)
        lgb_params, tuning_log = tune_hyperparameters(
            train_set, selected_features,
            n_folds=tune_cv_folds,
            n_trials=tune_n_trials,
        )
        _save(tuning_log, "06b_tuning_log.pkl")
        _save(lgb_params, "06c_best_params.pkl")
    elif do_tune and lgb_params is not None:
        print("\n  Skipping tuning: lgb_params already provided.")
    else:
        print("\n  Skipping tuning (do_tune=False). Using default params.")

    # ── STEP 6: Train final model ──
    print("\n" + "=" * 60)
    print(f"STEP 6: Training LightGBM ({len(selected_features)} features)")
    print("=" * 60)
    model = train_lightgbm(
        train_set, val_set,
        feature_cols=selected_features,
        lgb_params=lgb_params,
    )
    _save(model, "07_lgbm_model.txt")

    # ── STEP 7: Evaluation ──
    print("\n" + "=" * 60)
    print("STEP 7: Evaluation")
    print("=" * 60)
    val_preds = predict(model, val_set, feature_cols=selected_features)
    evaluate_median_ape(val_set, val_preds, group_col=group_col)
    tin_metrics = evaluate_by_tin(val_set, val_preds)
    val_full = val_set.assign(predicted=val_preds)
    val_full["ape"] = compute_ape(val_full["target"].values, val_full["predicted"].values)
    _save(val_full, "08_val_predictions.pkl")

    # ── STEP 8: Feature importance ──
    print("\n" + "=" * 60)
    print("STEP 8: Feature Importance (selected features)")
    print("=" * 60)
    imp_df = show_feature_importance(model)
    _save(imp_df, "09_feature_importance.pkl")

    # ── STEP 9: Forecast ──
    print("\n" + "=" * 60)
    print("STEP 9: Generating final forecasts")
    print("=" * 60)
    inference_data = build_inference_data(df)
    _save(inference_data, "10_inference_features.pkl")
    inference_data["forecast_12m_sum"] = predict(
        model, inference_data, feature_cols=selected_features
    )
    forecasts = inference_data[["tin", "origin_date", "forecast_12m_sum"]].copy()
    print(f"  Generated forecasts for {len(forecasts)} tins")
    _save(forecasts, "11_forecasts.pkl")

    if output_dir:
        print(f"\n  All artifacts saved to: {output_dir}/")

    # ── Store detailed results ──
    run_pipeline.last_results = {
        "model": model,
        "selected_features": selected_features,
        "selection_report": selection_report,
        "tuning_log": tuning_log,
        "lgb_params_used": lgb_params,
        "feature_importance": imp_df,
        "val_tin_metrics": tin_metrics,
        "val_set_full": val_full,
        "inference_features": inference_data,
    }

    return forecasts


# =============================================================================
# 9. USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data with various real-world patterns
    tins = [f"TIN_{i:05d}" for i in range(200)]
    rows = []
    for i, tin in enumerate(tins):
        tenure = np.random.randint(18, 61)
        start = pd.Timestamp("2019-01-01") + pd.DateOffset(months=np.random.randint(0, 24))
        dates = pd.date_range(start, periods=tenure, freq="MS")

        base = np.random.uniform(1000, 50000)
        trend = np.random.uniform(-100, 200)
        seasonal_amp = base * np.random.uniform(0, 0.3)
        seasonal_phase = np.random.randint(0, 12)

        values = []
        for t in range(tenure):
            month_of_year = (dates[t].month + seasonal_phase) % 12
            val = (
                base + trend * t
                + seasonal_amp * np.sin(2 * np.pi * month_of_year / 12)
                + np.random.normal(0, base * 0.1)
            )
            # Sparse tins (every 5th)
            if i % 5 == 0 and np.random.random() < 0.15:
                val = 0.0
            # Outlier tins (every 7th)
            if i % 7 == 0 and np.random.random() < 0.05:
                val = val * np.random.choice([3.0, 0.1])
            # Regime change tins (every 10th, halfway through)
            if i % 10 == 0 and t == tenure // 2:
                base = base * np.random.choice([1.5, 0.5])
            values.append(max(val, 0))

        for d, v in zip(dates, values):
            rows.append({"tin": tin, "ds": d, "y": v})

    sample_df = pd.DataFrame(rows)

    forecasts = run_pipeline(
        sample_df,
        min_history_months=6,
        val_months=12,
        max_origins_per_tin=24,
        # Feature selection options:
        do_feature_selection=True,
        corr_threshold=0.95,
        min_importance_pct=0.001,
        top_n_features=None,       # None = keep all that pass, or set e.g. 80
    )

    print("\n\nFinal output (DataFrame):")
    print(forecasts.head(10).to_string(index=False))
    print(f"\nShape: {forecasts.shape}")
    print(f"Columns: {list(forecasts.columns)}")

    # Access detailed results if needed:
    details = run_pipeline.last_results
    print(f"\nSelected features: {len(details['selected_features'])}")
    print(f"\nSelection report sample:")
    print(details["selection_report"].head(15).to_string(index=False))
