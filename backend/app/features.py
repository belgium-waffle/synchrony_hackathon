"""Stream A preprocessing.

Aguiar's `application_train_test()`, split into a fit/transform pair so a single
applicant can be scored at inference time, plus three additive feature blocks:

  1. Recency-windowed installment behaviour (30/90/180-day DPD and underpayment).
  2. EXT_SOURCE interaction terms (mean/max/min/var/product, NaN-safe).
  3. TOTAL_NULLS — a thin-file proxy counted on the raw, pre-cleaning row.

`ORGANIZATION_TYPE` and `OCCUPATION_TYPE` are deliberately left un-encoded here.
They are high-cardinality categoricals; one-hot would blow up the feature space and
plain frequency/label encoding invites leakage, so `scoring.py` owns a strict
out-of-fold target encoder for them instead. This module hands them back as clean
string columns ("passthrough") so the two concerns stay in the modules the task
assigns them to.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
TEXT_COL = "UNDERWRITER_NOTES"

BINARY_FEATURES = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
DAYS_EMPLOYED_ANOMALY = 365243

# Owned by scoring.py's KFoldTargetEncoder. Never one-hot encoded here.
HIGH_CARDINALITY_COLS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]

EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# (label -> lookback horizon in days). DAYS_INSTALMENT is negative, so "within the
# last 30 days" is DAYS_INSTALMENT >= -30.
INSTALLMENT_WINDOWS: Dict[str, int] = {"30D": 30, "90D": 90, "180D": 180}
INSTALLMENT_REQUIRED_COLS = {
    "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
}


def _is_text_dtype(series: pd.Series) -> bool:
    """Aguiar tested `dtype == object`; pandas 3.0 reports string columns as `str`."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True):
    original = list(df.columns)
    categorical = [c for c in df.columns if _is_text_dtype(df[c])]
    df = pd.get_dummies(df, columns=categorical, dummy_na=nan_as_category)
    return df, [c for c in df.columns if c not in original]


class AguiarFeatureEngine:
    RATIO_FEATURES = [
        "DAYS_EMPLOYED_PERC", "INCOME_CREDIT_PERC", "INCOME_PER_PERSON",
        "ANNUITY_INCOME_PERC", "PAYMENT_RATE", "CREDIT_GOODS_PERC",
    ]
    EXT_SOURCE_INTERACTION_FEATURES = [
        "EXT_SOURCE_MEAN", "EXT_SOURCE_MAX", "EXT_SOURCE_MIN",
        "EXT_SOURCE_VAR", "EXT_SOURCE_PRODUCT",
    ]
    INSTALLMENT_FEATURES = [
        f"INSTAL_{metric}_{agg}_{label}"
        for label in INSTALLMENT_WINDOWS
        for metric in ("DPD", "UNDERPAY")
        for agg in ("MEAN", "MAX")
    ] + ["HAS_INSTALLMENT_HISTORY"]

    def __init__(self, nan_as_category: bool = True):
        self.nan_as_category = nan_as_category
        self.factorize_maps_: Dict[str, Dict[Any, int]] = {}
        self.columns_: List[str] = []
        self.passthrough_categorical_cols_: List[str] = list(HIGH_CARDINALITY_COLS)

    # ------------------------------------------------------------ base steps
    @staticmethod
    def _total_nulls(df: pd.DataFrame) -> pd.Series:
        """Thin-file proxy. Counted on the RAW submitted row — before the
        DAYS_EMPLOYED sentinel is converted to NaN — so it reflects genuine
        missingness in what the applicant provided, not an artifact of our own
        cleaning step manufacturing additional NaNs downstream."""
        raw_cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
        return df[raw_cols].isna().sum(axis=1).astype(float)

    @staticmethod
    def _clean_and_ratio(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Aguiar: NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)
        # Aguiar's five domain ratios, verbatim
        df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
        df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
        df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
        # Added negative equity / risk domain ratio
        df["CREDIT_GOODS_PERC"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
        return df

    @staticmethod
    def _add_ext_source_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """Row-wise mean/max/min/var/product of the three EXT_SOURCE columns.

        numpy's `nan*` reductions already skip NaN correctly for mean/max/min. Two
        edge cases need an explicit override because numpy's identity-element
        convention would otherwise silently misrepresent "no data":
          - variance of a single known value is mathematically 0, but reporting
            that as "these sources agree perfectly" when we only observed one of
            them is misleading, so we require >=2 non-null sources.
          - `nanprod` treats an all-NaN row as an empty product (= 1), which would
            read as three unusually-agreeing zero-ish scores. Force NaN instead.
        """
        df = df.copy()
        vals = df[EXT_SOURCE_COLS].to_numpy(dtype=float)
        count = np.sum(~np.isnan(vals), axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
            df["EXT_SOURCE_MEAN"] = np.nanmean(vals, axis=1)
            df["EXT_SOURCE_MAX"] = np.nanmax(vals, axis=1)
            df["EXT_SOURCE_MIN"] = np.nanmin(vals, axis=1)
            variance = np.nanvar(vals, axis=1)
            product = np.nanprod(vals, axis=1)

        variance[count < 2] = np.nan
        product[count == 0] = np.nan
        df["EXT_SOURCE_VAR"] = variance
        df["EXT_SOURCE_PRODUCT"] = product
        return df

    def _join_installment_features(self, df: pd.DataFrame,
                                   installments_df: pd.DataFrame) -> pd.DataFrame:
        """Recency-windowed DPD and underpayment, joined by SK_ID_CURR.

        Uses `.map()` against a Series indexed by SK_ID_CURR rather than a merge,
        so `df`'s row order and index are never touched — required because Stream A
        later reads `y = engineered[TARGET]` off this same, order-preserved frame.
        """
        missing = INSTALLMENT_REQUIRED_COLS - set(installments_df.columns)
        if missing:
            raise ValueError(f"installments_df is missing required columns: {sorted(missing)}")

        df = df.copy()
        ins = installments_df.copy()
        if len(ins):
            # Aguiar's convention: DPD = actual payment day - scheduled day, floored
            # at 0 (an early or on-time payment is not "negative lateness").
            ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
            denom = ins["AMT_INSTALMENT"].replace(0, np.nan)
            ins["UNDERPAY_FRAC"] = ((ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]) / denom
                                    ).clip(lower=0, upper=1)

        has_history = ins.groupby("SK_ID_CURR").size() if len(ins) else pd.Series(dtype=float)

        for label, horizon in INSTALLMENT_WINDOWS.items():
            window = ins[ins["DAYS_INSTALMENT"] >= -horizon] if len(ins) else ins
            grouped = window.groupby("SK_ID_CURR") if len(window) else None
            for metric_col, out_name in (("DPD", "DPD"), ("UNDERPAY_FRAC", "UNDERPAY")):
                for agg in ("mean", "max"):
                    col_name = f"INSTAL_{out_name}_{agg.upper()}_{label}"
                    stat = grouped[metric_col].agg(agg) if grouped is not None else pd.Series(dtype=float)
                    df[col_name] = df[ID_COL].map(stat).fillna(0.0)

        df["HAS_INSTALLMENT_HISTORY"] = (df[ID_COL].map(has_history).fillna(0.0) > 0).astype(float)
        return df

    def _add_installment_features(self, df: pd.DataFrame,
                                  installments_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """A first-time applicant with genuinely zero prior installments and a
        caller that never wires in the table at all produce identical output: an
        empty-but-correctly-shaped installments_df takes the same code path as a
        populated one, rather than needing a separate "no data" branch to keep in
        sync with the real one."""
        if installments_df is None:
            installments_df = pd.DataFrame(columns=list(INSTALLMENT_REQUIRED_COLS))
        return self._join_installment_features(df, installments_df)

    def _extract_passthrough(self, df: pd.DataFrame):
        """Pull the high-cardinality categoricals out before one-hot encoding runs,
        so `one_hot_encoder` (which fires on every object-dtype column) never sees
        them. "XNA" is Home Credit's own sentinel for a missing/inapplicable
        category (e.g. an unemployed applicant's ORGANIZATION_TYPE) — reusing it
        keeps this consistent with the source data's own convention."""
        df = df.copy()
        for col in self.passthrough_categorical_cols_:
            if col not in df.columns:
                df[col] = "XNA"
            df[col] = df[col].fillna("XNA").astype(str)
        passthrough = df[self.passthrough_categorical_cols_].copy()
        df = df.drop(columns=self.passthrough_categorical_cols_)
        return df, passthrough

    # ------------------------------------------------------------- fit/transform
    def fit_transform(self, df: pd.DataFrame,
                      installments_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = df[df["CODE_GENDER"] != "XNA"].copy()          # Aguiar drops the XNA rows
        df = df.drop(columns=[TEXT_COL], errors="ignore")   # notes feed Stream B only
        df["TOTAL_NULLS"] = self._total_nulls(df)

        df = self._clean_and_ratio(df)
        df = self._add_ext_source_interactions(df)
        df = self._add_installment_features(df, installments_df)

        df, passthrough = self._extract_passthrough(df)

        for col in BINARY_FEATURES:                         # Aguiar: pd.factorize
            codes, uniques = pd.factorize(df[col])
            self.factorize_maps_[col] = {v: i for i, v in enumerate(uniques)}
            df[col] = codes

        df, _ = one_hot_encoder(df, self.nan_as_category)
        self.columns_ = [c for c in df.columns if c not in (ID_COL, TARGET)]

        # Same index throughout (no filter/reorder since the XNA drop above), so a
        # plain axis=1 concat is safe and avoids merge/reindex edge cases.
        return pd.concat([df, passthrough], axis=1)

    def transform(self, df: pd.DataFrame,
                 installments_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = df.drop(columns=[TEXT_COL], errors="ignore")
        df["TOTAL_NULLS"] = self._total_nulls(df)

        df = self._clean_and_ratio(df)
        df = self._add_ext_source_interactions(df)
        df = self._add_installment_features(df, installments_df)

        df, passthrough = self._extract_passthrough(df)

        for col in BINARY_FEATURES:
            df[col] = df[col].map(self.factorize_maps_[col]).fillna(-1).astype(int)
        df, _ = one_hot_encoder(df, self.nan_as_category)
        df = df.reindex(columns=self.columns_, fill_value=0)

        passthrough = passthrough.reset_index(drop=True)
        df = df.reset_index(drop=True)
        return pd.concat([df, passthrough], axis=1)


# ---------------------------------------------------------------------------
# Development data. Replace with application_train.csv / installments_payments.csv
# in staging/prod. Correlated deliberately (see docstrings below) so OOF AUC on
# this synthetic set is a meaningful smoke test of the new feature blocks — it is
# NOT a stand-in for validating against the real Home Credit competition data.
# ---------------------------------------------------------------------------
class MockApplicationData:
    EDUCATION = ["Secondary / secondary special", "Higher education",
                 "Incomplete higher", "Lower secondary"]
    INCOME_TYPE = ["Working", "Commercial associate", "State servant", "Pensioner"]
    FAMILY = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
    HOUSING = ["House / apartment", "With parents", "Rented apartment", "Municipal apartment"]
    OCCUPATION = ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers"]
    # A handful of Aguiar's real ORGANIZATION_TYPE categories, each given a small,
    # fixed risk offset so target encoding has genuine (if synthetic) signal to learn.
    ORGANIZATION_RISK: Dict[str, float] = {
        "Business Entity Type 3": 0.00, "Self-employed": 0.30, "Government": -0.20,
        "School": -0.15, "Trade: type 7": 0.35, "Military": -0.30, "Bank": -0.10,
        "Construction": 0.20, "Kindergarten": -0.10, "XNA": 0.10,
    }

    def __init__(self, n_rows: int = 2000, seed: int = 1001):
        self.n_rows, self.rng = n_rows, np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        n, rng = self.n_rows, self.rng
        amt_income_total = np.round(rng.lognormal(11.9, 0.50, n), -3).clip(25_000, 4_000_000)
        amt_credit = np.round(amt_income_total * rng.uniform(1.2, 8.0, n), -3)
        amt_goods_price = np.round(amt_credit * rng.uniform(0.80, 1.00, n), -3)
        cnt_payment = rng.integers(12, 60, n)
        amt_annuity = np.round(amt_credit / cnt_payment * rng.uniform(1.0, 1.25, n), -1)
        cnt_fam = rng.choice([1, 2, 3, 4, 5], n, p=[.22, .34, .24, .15, .05])

        days_birth = -rng.integers(7_500, 25_200, n)
        days_employed = -rng.gamma(2.0, 900, n).clip(0, 17_000).round(0)
        anom = rng.uniform(size=n) < 0.18
        days_employed = np.where(anom, DAYS_EMPLOYED_ANOMALY, days_employed)
        days_registration = -rng.gamma(2.2, 2_000, n).clip(0, 24_000).round(0)
        days_id_publish = -rng.integers(0, 6_000, n).astype(float)
        days_last_phone = -rng.gamma(1.6, 500, n).clip(0, 4_000).round(0)

        ext1, ext2, ext3 = (rng.beta(2.6, 2.6, n).round(4),
                            rng.beta(3.0, 2.2, n).round(4),
                            rng.beta(2.4, 2.4, n).round(4))
        education = rng.choice(self.EDUCATION, n, p=[.71, .24, .03, .02])
        region_rating = rng.choice([1, 2, 3], n, p=[.15, .74, .11])
        bureau_year = rng.poisson(1.9, n).clip(0, 25).astype(float)
        obs30 = rng.poisson(1.4, n).clip(0, 20).astype(float)
        def30 = rng.poisson(0.15, n).clip(0, 8).astype(float)

        org_types = list(self.ORGANIZATION_RISK.keys())
        organization_type = np.where(anom, "XNA",
                                     rng.choice(org_types[:-1], n))  # non-XNA if employed

        annuity_income = amt_annuity / amt_income_total
        payment_rate = amt_annuity / amt_credit
        org_offset = np.array([self.ORGANIZATION_RISK[o] for o in organization_type])
        z = (-1.55
             - 1.05 * self._z(ext2) - 1.15 * self._z(ext3) - 0.70 * self._z(ext1)
             - 0.55 * self._z(-days_birth) - 0.30 * self._z(np.log(amt_income_total))
             + 0.60 * self._z(annuity_income) + 0.45 * self._z(payment_rate)
             + 0.30 * self._z(cnt_fam) + 0.35 * self._z(bureau_year) + 0.40 * self._z(def30)
             - 0.25 * self._z(-days_registration) + 0.45 * anom.astype(float)
             + np.where(education == "Higher education", -0.45, 0.10)
             + org_offset
             + rng.normal(0, 0.40, n))
        target = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(int)

        # Latent "payment discipline" correlated with true risk (z) plus independent
        # noise, so installment behaviour is a genuine, non-circular predictor —
        # correlated with the label, not a re-encoding of it.
        self._payment_risk_factor = self._z(z) * 0.65 + rng.normal(0, 0.85, n)

        df = pd.DataFrame({
            ID_COL: np.arange(100_001, 100_001 + n), TARGET: target,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[.90, .10]),
            "CODE_GENDER": rng.choice(["F", "M"], n, p=[.66, .34]),
            "FLAG_OWN_CAR": rng.choice(["N", "Y"], n, p=[.66, .34]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[.69, .31]),
            "CNT_FAM_MEMBERS": cnt_fam.astype(float),
            "AMT_INCOME_TOTAL": amt_income_total, "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity, "AMT_GOODS_PRICE": amt_goods_price,
            "NAME_INCOME_TYPE": rng.choice(self.INCOME_TYPE, n, p=[.52, .23, .07, .18]),
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": rng.choice(self.FAMILY, n, p=[.64, .15, .10, .06, .05]),
            "NAME_HOUSING_TYPE": rng.choice(self.HOUSING, n, p=[.89, .05, .04, .02]),
            "OCCUPATION_TYPE": rng.choice(self.OCCUPATION, n),
            "ORGANIZATION_TYPE": organization_type,
            "REGION_RATING_CLIENT": region_rating,
            "DAYS_BIRTH": days_birth.astype(float), "DAYS_EMPLOYED": days_employed.astype(float),
            "DAYS_REGISTRATION": days_registration, "DAYS_ID_PUBLISH": days_id_publish,
            "DAYS_LAST_PHONE_CHANGE": days_last_phone,
            "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
            "OBS_30_CNT_SOCIAL_CIRCLE": obs30, "DEF_30_CNT_SOCIAL_CIRCLE": def30,
            "AMT_REQ_CREDIT_BUREAU_YEAR": bureau_year,
        })
        for col, frac in [("EXT_SOURCE_1", .50), ("EXT_SOURCE_3", .20),
                          ("OCCUPATION_TYPE", .31), ("AMT_ANNUITY", .02)]:
            idx = self.rng.choice(df.index, size=int(len(df) * frac), replace=False)
            df.loc[idx, col] = np.nan
        return df

    def generate_installments(self, application_df: pd.DataFrame) -> pd.DataFrame:
        """Synthetic `installments_payments.csv`, correlated with the applicant's
        latent payment-discipline factor set in `generate()`. Requires `generate()`
        to have been called first on the same instance."""
        if not hasattr(self, "_payment_risk_factor"):
            raise RuntimeError("Call generate() before generate_installments().")

        rng = self.rng
        risk = self._payment_risk_factor
        prob_late = 1.0 / (1.0 + np.exp(-(risk - 1.0)))    # per-installment lateness odds
        rows = []
        for i, sk_id in enumerate(application_df[ID_COL].to_numpy()):
            n_installments = rng.poisson(6) + 1
            annuity = float(application_df.iloc[i]["AMT_ANNUITY"]) or 15_000.0
            if np.isnan(annuity) or annuity <= 0:
                annuity = 15_000.0
            scheduled_days = rng.choice(np.arange(-720, -10), size=n_installments, replace=False)
            for day in scheduled_days:
                is_late = rng.uniform() < prob_late[i]
                dpd = max(0, int(rng.normal(18, 8))) if is_late else 0
                entry_day = min(day + dpd, -1)
                underpay = max(0.0, rng.normal(0.08, 0.05)) if (is_late and rng.uniform() < 0.4) else 0.0
                amt_instalment = round(annuity * rng.uniform(0.9, 1.1), 2)
                amt_payment = round(amt_instalment * (1 - min(underpay, 1.0)), 2)
                rows.append((sk_id, int(day), int(entry_day), amt_instalment, amt_payment))

        return pd.DataFrame(rows, columns=[
            "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
        ])

    @staticmethod
    def _z(x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)