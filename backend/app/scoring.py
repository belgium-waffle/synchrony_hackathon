"""Stream A — deterministic quantitative scoring.

Two additions over the single-model baseline, both driven by the same design
constraint: the Deterministic Verifier checks TreeSHAP numbers to 1e-3, so nothing
here is allowed to be merely "close."

1. `KFoldTargetEncoder` — leak-safe out-of-fold target encoding for the two
   high-cardinality categoricals (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`) that
   `features.py` deliberately leaves un-encoded.

2. `DualModelScoringEngine` — LightGBM + CatBoost trained per fold on an identical
   numeric feature matrix, blended in MARGIN (log-odds) space rather than by rank
   or raw probability. Margin averaging is the one blend that keeps the SHAP
   decomposition exact: TreeSHAP guarantees `margin = base + sum(shap)` for each
   model individually, so averaging both sides of that equation for two models
   gives `blended_margin = blended_base + sum(blended_shap)` for free, with no
   approximation. Rank averaging — a common Kaggle ensembling trick — was
   considered and rejected for this system: it operates on order statistics, which
   have no additive relationship to either model's SHAP output, so there is no way
   to make a rank-averaged blend satisfy the verifier's exact-sum requirement.
"""
from __future__ import annotations

import gc
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import HIGH_CARDINALITY_COLS, ID_COL, TARGET, AguiarFeatureEngine

log = logging.getLogger(__name__)

MODEL_VERSION = "stream-a-lgbm-catboost-ensemble-2.0.0"

# Aguiar's Bayesian-optimised LightGBM parameters, reproduced exactly.
# Source: Tilii's kernel, via "LightGBM with Simple Features".
AGUIAR_LGBM_PARAMS: Dict[str, Any] = {
    "nthread": 4, "n_estimators": 10000, "learning_rate": 0.02, "num_leaves": 34,
    "colsample_bytree": 0.9497036, "subsample": 0.8715623, "max_depth": 8,
    "reg_alpha": 0.041545473, "reg_lambda": 0.0735294, "min_split_gain": 0.0222415,
    "min_child_weight": 39.3259775, "verbose": -1,
}
AGUIAR_TRAIN_ROWS = 307_511

CATBOOST_PARAMS: Dict[str, Any] = {
    "iterations": 4000, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3.0,
    "rsm": 0.8, "random_strength": 1.5,
    "loss_function": "Logloss", "eval_metric": "AUC", "verbose": False,
    "allow_writing_files": False, "od_type": "Iter", "od_wait": 200,
}


def lgbm_params_for_sample(n_rows: int) -> Dict[str, Any]:
    """Aguiar's params, rescaled for a small training sample.

    `min_child_weight` is `min_sum_hessian_in_leaf`. For binary logloss the per-row
    hessian is p(1-p) <= 0.25, so 39.33 demands roughly 160+ rows per leaf. On the
    full 307k rows that is a sensible regulariser; on a small development sample it
    blocks every candidate split, LightGBM returns a constant model, and every SHAP
    value is exactly zero. Scaling by the sample-size ratio is a no-op on the real
    dataset.
    """
    params = dict(AGUIAR_LGBM_PARAMS)
    ratio = n_rows / AGUIAR_TRAIN_ROWS
    if ratio < 1.0:
        params["min_child_weight"] = round(max(AGUIAR_LGBM_PARAMS["min_child_weight"] * ratio, 1.0), 6)
        params["n_estimators"] = 2000
        log.warning(
            "LightGBM sample-size guard active: %s rows vs %s tuned rows. "
            "min_child_weight %s -> %s, n_estimators %s -> %s.",
            f"{n_rows:,}", f"{AGUIAR_TRAIN_ROWS:,}", AGUIAR_LGBM_PARAMS["min_child_weight"],
            params["min_child_weight"], AGUIAR_LGBM_PARAMS["n_estimators"], params["n_estimators"],
        )
    return params


# ---------------------------------------------------------------------------
# Out-of-fold target encoding
# ---------------------------------------------------------------------------
class KFoldTargetEncoder:
    """Leak-safe out-of-fold target encoding for high-cardinality categoricals.

    Two independent fold counts are used deliberately. `n_encoding_folds` (default
    10) is decoupled from the scoring engine's own `n_model_folds` (default 5). The
    property that actually prevents a row's label leaking into its own feature is
    that its OOF-encoded value is computed excluding its own encoding-fold — full
    stop, regardless of what those folds are used for elsewhere. Using MORE, SMALLER
    encoding folds than model folds shrinks a real but subtler leakage channel: with
    a single shared 5-fold split, a training row's encoding draws on the other 4
    folds, one of which is that particular model-fold's own validation set, letting
    a sliver of that fold's label distribution leak into the training data through
    the encoding. Ten encoding folds cut that overlap roughly in half per row. This
    does not reach the guarantee of fully nested cross-validation (no fixed-size
    K-fold scheme does), but it is the standard, much cheaper compromise used in
    practice, and is paired with Bayesian mean-shrinkage smoothing below so that
    rare categories fall back toward the global rate rather than overfitting to a
    handful of rows.
    """

    def __init__(self, columns: List[str], n_encoding_folds: int = 10,
                smoothing: float = 20.0, seed: int = 1001):
        self.columns = list(columns)
        self.n_encoding_folds = n_encoding_folds
        self.smoothing = smoothing
        self.seed = seed
        self.global_mean_: Optional[float] = None
        # Per-column, fitted on ALL training rows — used only at inference. Never
        # used to encode a training row, which is what would reintroduce leakage.
        self.full_maps_: Dict[str, pd.Series] = {}

    def _smoothed_means(self, values: pd.Series, y: np.ndarray, global_mean: float) -> pd.Series:
        frame = pd.DataFrame({"cat": values.to_numpy(), "y": y})
        stats = frame.groupby("cat")["y"].agg(["mean", "count"])
        return (stats["count"] * stats["mean"] + self.smoothing * global_mean) / (
            stats["count"] + self.smoothing
        )

    def fit_transform_oof(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Returns a new DataFrame of `<col>_TE` columns, out-of-fold encoded for
        training, and — as a side effect — fits the full-data maps used later by
        `transform()` at inference."""
        self.global_mean_ = float(np.mean(y))
        n = len(df)
        oof = pd.DataFrame(index=df.index)

        folds = StratifiedKFold(self.n_encoding_folds, shuffle=True, random_state=self.seed)
        for col in self.columns:
            encoded = np.full(n, self.global_mean_, dtype=float)
            for train_idx, holdout_idx in folds.split(df, y):
                fold_means = self._smoothed_means(
                    df[col].iloc[train_idx], y[train_idx], self.global_mean_
                )
                encoded[holdout_idx] = (
                    df[col].iloc[holdout_idx].map(fold_means).fillna(self.global_mean_).to_numpy()
                )
            oof[f"{col}_TE"] = encoded

        self.fit_full(df, y)
        return oof

    def fit_full(self, df: pd.DataFrame, y: np.ndarray) -> "KFoldTargetEncoder":
        """Fit on every training row, with no held-out split. Used exclusively for
        transforming new (inference-time) rows — never for encoding a training row,
        which is exactly the leakage this class exists to prevent."""
        self.global_mean_ = float(np.mean(y))
        for col in self.columns:
            self.full_maps_[col] = self._smoothed_means(df[col], y, self.global_mean_)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None:
            raise RuntimeError("Call fit_transform_oof() or fit_full() before transform().")
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            out[f"{col}_TE"] = df[col].map(self.full_maps_[col]).fillna(self.global_mean_)
        return out


# ---------------------------------------------------------------------------
# Feature families — keeps the LLM's working set small and maps onto ECOA/Reg B
# adverse-action reason codes. Raw-column granularity would flood the prompt.
# ---------------------------------------------------------------------------
FEATURE_FAMILIES: Dict[str, List[str]] = {
    "External Scores": [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "EXT_SOURCE_MEAN", "EXT_SOURCE_MAX", "EXT_SOURCE_MIN",
        "EXT_SOURCE_VAR", "EXT_SOURCE_PRODUCT",
    ],
    "Debt-to-Income": [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "CNT_FAM_MEMBERS", "INCOME_CREDIT_PERC", "INCOME_PER_PERSON",
        "ANNUITY_INCOME_PERC", "PAYMENT_RATE", "NAME_CONTRACT_TYPE",
        "CREDIT_GOODS_PERC",
    ],
    "Employment History": [
        "DAYS_EMPLOYED", "DAYS_EMPLOYED_PERC", "NAME_INCOME_TYPE", "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
    ],
    "Demographics & Stability": [
        "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
        "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "REGION_RATING_CLIENT",
    ],
    "Bureau & Social Circle": [
        "AMT_REQ_CREDIT_BUREAU_YEAR", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    ],
    "Recent Payment Behavior": [
        *AguiarFeatureEngine.INSTALLMENT_FEATURES,
    ],
    "Data Completeness": ["TOTAL_NULLS"],
}
_BASE_TO_FAMILY = {b: f for f, bases in FEATURE_FAMILIES.items() for b in bases}
# Longest first: DAYS_EMPLOYED_PERC before DAYS_EMPLOYED, ORGANIZATION_TYPE_TE
# before... well there's only ORGANIZATION_TYPE, but the rule generalises safely.
_SORTED_BASES = sorted(_BASE_TO_FAMILY, key=len, reverse=True)


def resolve_family(column: str) -> str:
    """Map an engineered column to its business family.

    A target-encoded column such as "OCCUPATION_TYPE_TE" is matched by the same
    longest-prefix rule that already handles one-hot suffixes ("OCCUPATION_TYPE_
    Laborers") — no special-casing needed for the new encoding scheme.
    """
    if column in _BASE_TO_FAMILY:
        return _BASE_TO_FAMILY[column]
    for base in _SORTED_BASES:
        if column.startswith(base + "_"):
            return _BASE_TO_FAMILY[base]
    return "Applicant Profile"

@dataclass
class FamilyAttribution:
    family: str
    contribution: float
    contribution_pct: float
    direction: str
    top_features: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class QuantResult:
    applicant_id: str
    pd_probability: float
    risk_score: int
    base_value: float
    family_attributions: List[FamilyAttribution]
    raw_shap: Dict[str, float] = field(default_factory=dict)
    model_margins: Dict[str, float] = field(default_factory=dict)  # diagnostic only

    def families_dict(self) -> Dict[str, FamilyAttribution]:
        return {fa.family: fa for fa in self.family_attributions}


class DualModelScoringEngine:
    """LightGBM + CatBoost, fold-bagged, blended in margin space.

    Both models train on the SAME numeric design matrix (one-hot + target-encoded
    categoricals — CatBoost gets no `cat_features` and therefore no native
    categorical handling). That is a deliberate trade: CatBoost's own ordered
    target statistics would put its SHAP attributions in a different feature
    representation than LightGBM's one-hot columns, and reconciling the two exactly
    would be substantially harder than the accuracy CatBoost's native handling
    would otherwise buy here.
    """
    SCORE_MIN, SCORE_MAX = 300, 850
    SHAP_SELF_CHECK_TOL = 1e-6      # tight — this is machine precision, not the verifier's 1e-3

    def __init__(self, n_folds: int = 5, n_encoding_folds: int = 10, seed: int = 1001):
        self.n_folds, self.seed = n_folds, seed
        self.fe = AguiarFeatureEngine()
        self.target_encoder = KFoldTargetEncoder(HIGH_CARDINALITY_COLS,
                                                 n_encoding_folds=n_encoding_folds, seed=seed)
        self.lgbm_models_: List[LGBMClassifier] = []
        self.cat_models_: List[CatBoostClassifier] = []
        self.lgbm_explainers_: List[shap.TreeExplainer] = []
        self.cat_explainers_: List[shap.TreeExplainer] = []
        self.feature_names_: List[str] = []
        self.oof_auc_lgbm_: Optional[float] = None
        self.oof_auc_cat_: Optional[float] = None
        self.oof_auc_blend_: Optional[float] = None
        self.fold_aucs_blend_: List[float] = []

    # ------------------------------------------------------------- training
    def fit(self, raw_df: pd.DataFrame, installments_df: Optional[pd.DataFrame] = None
           ) -> "DualModelScoringEngine":
        engineered = self.fe.fit_transform(raw_df, installments_df)
        y = engineered[TARGET].to_numpy()

        categorical_raw = engineered[HIGH_CARDINALITY_COLS]
        numeric = engineered.drop(columns=[ID_COL, TARGET, *HIGH_CARDINALITY_COLS])

        te_oof = self.target_encoder.fit_transform_oof(categorical_raw, y)
        X = pd.concat([numeric.reset_index(drop=True), te_oof.reset_index(drop=True)], axis=1)
        
        # Clean special characters out of column names to prevent LightGBM JSON crash
        X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)) for c in X.columns]
        self.feature_names_ = X.columns.tolist()

        lgbm_params = lgbm_params_for_sample(len(X))
        folds = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
        oof_margin_lgbm = np.zeros(len(X))
        oof_margin_cat = np.zeros(len(X))

        for n_fold, (tr, va) in enumerate(folds.split(X, y)):
            lgbm = LGBMClassifier(random_state=self.seed, **lgbm_params)
            lgbm.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
                    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            self.lgbm_models_.append(lgbm)
            oof_margin_lgbm[va] = lgbm.predict(X.iloc[va], raw_score=True)

            cat = CatBoostClassifier(random_seed=self.seed, **CATBOOST_PARAMS)
            cat.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]), use_best_model=True)
            self.cat_models_.append(cat)
            oof_margin_cat[va] = cat.predict(X.iloc[va], prediction_type="RawFormulaVal")

            blended_margin_fold = 0.5 * oof_margin_lgbm[va] + 0.5 * oof_margin_cat[va]
            fold_auc = roc_auc_score(y[va], 1.0 / (1.0 + np.exp(-blended_margin_fold)))
            self.fold_aucs_blend_.append(fold_auc)
            log.info("Fold %d — LGBM trees: %d, CatBoost trees: %d, blended AUC: %.6f",
                     n_fold + 1, lgbm.booster_.num_trees(), cat.tree_count_, fold_auc)
            gc.collect()

        self.oof_auc_lgbm_ = roc_auc_score(y, 1.0 / (1.0 + np.exp(-oof_margin_lgbm)))
        self.oof_auc_cat_ = roc_auc_score(y, 1.0 / (1.0 + np.exp(-oof_margin_cat)))
        blended_oof = 0.5 * oof_margin_lgbm + 0.5 * oof_margin_cat
        self.oof_auc_blend_ = roc_auc_score(y, 1.0 / (1.0 + np.exp(-blended_oof)))
        log.info("OOF AUC — LightGBM: %.6f | CatBoost: %.6f | Blend: %.6f",
                 self.oof_auc_lgbm_, self.oof_auc_cat_, self.oof_auc_blend_)

        if sum(m.booster_.num_trees() for m in self.lgbm_models_) == 0:
            raise RuntimeError("LightGBM produced zero trees across all folds.")

        self.lgbm_explainers_ = [shap.TreeExplainer(m) for m in self.lgbm_models_]
        self.cat_explainers_ = [shap.TreeExplainer(m) for m in self.cat_models_]
        self._self_check_shap_additivity(X.iloc[:min(20, len(X))])
        return self

    def _self_check_shap_additivity(self, X_sample: pd.DataFrame) -> None:
        """Fail at fit time, loudly, rather than let a SHAP/library version
        mismatch silently ship numbers that the verifier will later reject one
        request at a time. Checks both models individually and the blend."""
        for lgbm, ex in zip(self.lgbm_models_, self.lgbm_explainers_):
            margin = lgbm.predict(X_sample, raw_score=True)
            sv = self._positive_class_shap(ex.shap_values(X_sample))
            ev = self._expected_value(ex)
            recon = ev + sv.sum(axis=1)
            if not np.allclose(recon, margin, atol=self.SHAP_SELF_CHECK_TOL):
                raise RuntimeError(
                    f"LightGBM SHAP additivity check failed: max diff "
                    f"{np.max(np.abs(recon - margin)):.2e}"
                )
        for cat, ex in zip(self.cat_models_, self.cat_explainers_):
            margin = cat.predict(X_sample, prediction_type="RawFormulaVal")
            sv = np.asarray(ex.shap_values(X_sample), dtype=float)
            ev = float(ex.expected_value)
            recon = ev + sv.sum(axis=1)
            if not np.allclose(recon, margin, atol=self.SHAP_SELF_CHECK_TOL):
                raise RuntimeError(
                    f"CatBoost SHAP additivity check failed: max diff "
                    f"{np.max(np.abs(recon - margin)):.2e}"
                )
        log.info("SHAP additivity self-check passed for both model families "
                 "(%d models, tol %.0e).", len(self.lgbm_models_) + len(self.cat_models_),
                 self.SHAP_SELF_CHECK_TOL)

    # ---------------------------------------------------------- explanation
    @staticmethod
    def _positive_class_shap(values: Any) -> np.ndarray:
        if isinstance(values, list):
            values = values[-1]
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 3:
            arr = arr[..., -1]
        return arr

    @staticmethod
    def _expected_value(explainer: shap.TreeExplainer) -> float:
        ev = explainer.expected_value
        return float(np.asarray(ev).ravel()[-1]) if isinstance(ev, (list, np.ndarray)) else float(ev)

    def _row_to_matrix(self, applicant: Dict[str, Any]) -> pd.DataFrame:
        row_df = pd.DataFrame([applicant])
        engineered = self.fe.transform(row_df)
        categorical_raw = engineered[HIGH_CARDINALITY_COLS]
        numeric = engineered.drop(columns=[c for c in (ID_COL, TARGET) if c in engineered]
                                  + HIGH_CARDINALITY_COLS, errors="ignore")
        te = self.target_encoder.transform(categorical_raw).reset_index(drop=True)
        X_row = pd.concat([numeric.reset_index(drop=True), te], axis=1)
        
        # Apply the exact same sanitizer logic to the inference row
        X_row.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)) for c in X_row.columns]
        
        return X_row.reindex(columns=self.feature_names_, fill_value=0.0)

    def _averaged_margin_and_shap(self, X_row: pd.DataFrame) -> Tuple[float, np.ndarray, float, Dict[str, float]]:
        lgbm_margins, lgbm_shaps, lgbm_bases = [], [], []
        for lgbm, ex in zip(self.lgbm_models_, self.lgbm_explainers_):
            lgbm_margins.append(float(lgbm.predict(X_row, raw_score=True)[0]))
            lgbm_shaps.append(self._positive_class_shap(ex.shap_values(X_row))[0])
            lgbm_bases.append(self._expected_value(ex))

        cat_margins, cat_shaps, cat_bases = [], [], []
        for cat, ex in zip(self.cat_models_, self.cat_explainers_):
            cat_margins.append(float(cat.predict(X_row, prediction_type="RawFormulaVal")[0]))
            cat_shaps.append(np.asarray(ex.shap_values(X_row), dtype=float)[0])
            cat_bases.append(float(ex.expected_value))

        lgbm_margin = float(np.mean(lgbm_margins))
        lgbm_shap = np.mean(lgbm_shaps, axis=0)
        lgbm_base = float(np.mean(lgbm_bases))

        cat_margin = float(np.mean(cat_margins))
        cat_shap = np.mean(cat_shaps, axis=0)
        cat_base = float(np.mean(cat_bases))

        # THE blend. 50/50 in margin space — see module docstring for why.
        blended_margin = 0.5 * lgbm_margin + 0.5 * cat_margin
        blended_shap = 0.5 * lgbm_shap + 0.5 * cat_shap
        blended_base = 0.5 * lgbm_base + 0.5 * cat_base

        # Re-verified on every single request, not just at fit time — this is the
        # exact invariant the Deterministic Verifier checks downstream, so Stream A
        # checks it first and fails loudly rather than hand the verifier a payload
        # it would have to block anyway.
        reconstructed = blended_base + float(blended_shap.sum())
        if abs(reconstructed - blended_margin) > 1e-6:
            raise RuntimeError(
                f"Blended SHAP does not reconstruct the blended margin: "
                f"{reconstructed:.8f} vs {blended_margin:.8f} "
                f"(diff {abs(reconstructed - blended_margin):.2e})"
            )

        diagnostics = {"lgbm_margin": lgbm_margin, "catboost_margin": cat_margin,
                       "blended_margin": blended_margin}
        return blended_margin, blended_shap, blended_base, diagnostics

    def _group_into_families(self, shap_row: np.ndarray) -> List[FamilyAttribution]:
        buckets: Dict[str, List[Tuple[str, float]]] = {f: [] for f in FEATURE_FAMILIES}
    
        # Safely assign unrecognized features to their fallback bucket
        for col, val in zip(self.feature_names_, shap_row):
            family = resolve_family(col)
            buckets.setdefault(family, []).append((col, float(val)))
            
        total_abs = sum(abs(v) for pairs in buckets.values() for _, v in pairs) or 1.0
        out = []
        for fam, pairs in buckets.items():
            contribution = float(sum(v for _, v in pairs))
            fam_abs = sum(abs(v) for _, v in pairs)
            top = sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)[:3]
            out.append(FamilyAttribution(
                family=fam, contribution=round(contribution, 4),
                contribution_pct=round(100.0 * fam_abs / total_abs, 1),
                direction="increases_risk" if contribution > 0 else "decreases_risk",
                top_features=[(c, round(v, 4)) for c, v in top],
            ))
        return sorted(out, key=lambda fa: fa.contribution_pct, reverse=True)

    # ------------------------------------------------------------ inference
    def predict_proba_row(self, applicant: Dict[str, Any]) -> float:
        X_row = self._row_to_matrix(applicant)
        margin, _, _, _ = self._averaged_margin_and_shap(X_row)
        return float(1.0 / (1.0 + np.exp(-margin)))

    def score_applicant(self, applicant: Dict[str, Any]) -> QuantResult:
        if not self.lgbm_models_ or not self.cat_models_:
            raise RuntimeError("Engine not fitted.")
        X_row = self._row_to_matrix(applicant)
        margin, shap_row, base, diagnostics = self._averaged_margin_and_shap(X_row)
        pd_prob = float(1.0 / (1.0 + np.exp(-margin)))
        return QuantResult(
            applicant_id=str(applicant.get(ID_COL, "UNKNOWN")),
            pd_probability=round(pd_prob, 6),
            risk_score=self.probability_to_score(pd_prob),
            base_value=round(base, 4),
            family_attributions=self._group_into_families(shap_row),
            raw_shap={c: round(float(v), 6) for c, v in zip(self.feature_names_, shap_row)},
            model_margins={k: round(v, 6) for k, v in diagnostics.items()},
        )

    @classmethod
    def probability_to_score(cls, pd_prob: float) -> int:
        span = cls.SCORE_MAX - cls.SCORE_MIN
        return int(round(cls.SCORE_MIN + (1.0 - float(pd_prob)) * span))


# Backward-compatible alias — main.py and recourse.py import LightGBMScoringEngine.
LightGBMScoringEngine = DualModelScoringEngine