from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os
import warnings
from datetime import datetime
import io
import contextlib


def get_hmm_fit_defaults(cfg=None):
    n_iter = int(getattr(cfg, "hmm_n_iter", 500)) if cfg is not None else 500
    tol = float(getattr(cfg, "hmm_tol", 1e-2)) if cfg is not None else 1e-2
    min_covar = float(getattr(cfg, "min_covar", 1e-3)) if cfg is not None else 1e-3
    return {"n_iter": n_iter, "tol": tol, "min_covar": min_covar}

def _force_random_init_hmm(model: GaussianHMM, X_fit_s: np.ndarray, seed: int, min_covar: float = 1e-3):
    if X_fit_s is None or len(X_fit_s) < model.n_components:
        return
    rng = np.random.default_rng(int(seed))
    k = int(model.n_components)
    n, d = X_fit_s.shape
    sp = rng.random(k)
    sp = sp / sp.sum()
    tm = rng.random((k, k))
    tm = tm / tm.sum(axis=1, keepdims=True)
    idx = rng.choice(n, size=k, replace=False)
    means = X_fit_s[idx].copy()
    cov = np.cov(X_fit_s, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + np.eye(d) * float(min_covar)
    covars = np.stack([cov.copy() for _ in range(k)], axis=0)
    model.startprob_ = sp
    model.transmat_ = tm
    model.means_ = means
    model.covars_ = covars
    model.init_params = ""


_HMM_WARN_SUMMARY_PRINTED = False


def _should_verbose_warnings() -> bool:
    v = os.getenv("QRS_VERBOSE_WARNINGS", "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _warn_log_path() -> str | None:
    return os.getenv("QRS_HMM_WARN_LOG", "").strip() or None


def _append_warn_log(messages: list[str]) -> None:
    if not messages:
        return
    path = _warn_log_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            for m in messages:
                f.write(f"[{stamp}] {m}\n")
    except Exception:
        return


def _run_with_warning_capture(fn):
    global _HMM_WARN_SUMMARY_PRINTED
    if _should_verbose_warnings():
        return fn()
    captured: list[str] = []

    def _capture(message, category, filename, lineno, file=None, line=None):
        captured.append(str(message))

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        old_show = warnings.showwarning
        warnings.showwarning = _capture
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                result = fn()
        finally:
            warnings.showwarning = old_show
    extra = (stdout_buf.getvalue() + "\n" + stderr_buf.getvalue()).strip().splitlines()
    for line in extra:
        line = line.strip()
        if not line:
            continue
        if "not converging" in line.lower():
            captured.append(line)
    if captured:
        _append_warn_log(captured)
        if not _HMM_WARN_SUMMARY_PRINTED:
            _HMM_WARN_SUMMARY_PRINTED = True
            path = _warn_log_path() or "logs/hmm_warnings.txt"
            print(f"[WARN] HMM not converging (count={len(captured)}). See {path}")
    return result


def align_states_by_volatility(model, X_sample=None):
    try:
        if model.covariance_type == "full":
            vols = np.array([np.mean(np.diag(c)) for c in model.covars_])
        elif model.covariance_type == "diag":
            vols = np.mean(model.covars_, axis=1)
        elif model.covariance_type == "spherical":
            vols = model.covars_.flatten()
        else:
            return

        sorted_idx = np.argsort(vols)

        if np.array_equal(sorted_idx, np.arange(len(vols))):
            return

        old_startprob = model.startprob_.copy()
        old_transmat = model.transmat_.copy()
        old_means = model.means_.copy()
        old_covars = model.covars_.copy()

        model.startprob_ = old_startprob[sorted_idx]
        model.transmat_ = old_transmat[sorted_idx][:, sorted_idx]
        model.means_ = old_means[sorted_idx]
        model.covars_ = old_covars[sorted_idx]

    except Exception:
        return


def fit_hmm_train_infer(
    feat_ready: pd.DataFrame,
    feature_cols: List[str],
    n_states: int,
    base_random_state: int,
    exclude_last_day_from_fit: bool,
    num_inits: int,
    force_random_init: bool = False,
    min_covar: float = 1e-3,
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
    infer_start_date: Optional[str] = None,
    infer_warmup_bars: int = 0,
) -> Tuple[GaussianHMM, StandardScaler, np.ndarray, float]:
    """
    Fit HMM on a restricted (historical) slice, then infer states.

    - Fit is controlled by pick_fit_mask_by_date(idx, ..., train_start_date/train_end_date).
    - If infer_start_date is provided: states for rows >= infer_start_date are inferred in an
      infer-only manner with an optional warmup prefix (infer_warmup_bars) to preserve continuity.
      Rows < infer_start_date are inferred on the historical slice only (no leakage from future).
    """
    if feat_ready is None or len(feat_ready) == 0:
        raise ValueError("feat_ready is empty")

    X_all = feat_ready[feature_cols].values.astype(float)

    # -------- Fit mask (historical only) --------
    from quant_refactor_skeleton.hmm.train_split import pick_fit_mask_by_date

    fit_mask = pick_fit_mask_by_date(
        feat_ready.index,
        exclude_last_day_from_fit,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
    )
    if fit_mask.sum() < max(200, int(n_states) * 50):
        fit_mask[:] = True

    X_fit = X_all[fit_mask]
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)

    best_model = None
    best_ll = -np.inf

    for i in range(int(num_inits)):
        current_seed = int(base_random_state) + i
        try:
            model = GaussianHMM(
                n_components=int(n_states),
                covariance_type="full",
                n_iter=500,
                random_state=current_seed,
                verbose=False,
                min_covar=float(min_covar),
            )
            if force_random_init:
                _force_random_init_hmm(model, X_fit_s, current_seed, min_covar=float(min_covar))
            _run_with_warning_capture(lambda: model.fit(X_fit_s))
        except ValueError:
            try:
                model = GaussianHMM(
                    n_components=int(n_states),
                    covariance_type="diag",
                    n_iter=500,
                    random_state=current_seed,
                    verbose=False,
                    min_covar=1e-2,
                )
                _run_with_warning_capture(lambda: model.fit(X_fit_s))
            except Exception:
                continue

        ll = model.score(X_fit_s)
        if ll > best_ll:
            best_ll = ll
            best_model = model

    if best_model is None:
        best_model = GaussianHMM(
            n_components=int(n_states),
            covariance_type="diag",
            n_iter=500,
            random_state=int(base_random_state),
            min_covar=0.1,
        )
        _run_with_warning_capture(lambda: best_model.fit(X_fit_s))
        best_ll = best_model.score(X_fit_s)

    # Align states for interpretability (low-vol -> low index, etc.)
    align_states_by_volatility(best_model)

    # -------- Inference --------
    X_all_s = scaler.transform(X_all)
    idx = feat_ready.index

    if not infer_start_date:
        states = best_model.predict(X_all_s)
        return best_model, scaler, states, float(best_ll)

    infer_start_dt = pd.to_datetime(infer_start_date)
    hist_mask = idx < infer_start_dt
    trade_mask = ~hist_mask

    states = np.full(len(idx), -1, dtype=int)

    # Infer historical part WITHOUT seeing trade part (no leakage)
    if hist_mask.any():
        states[hist_mask] = best_model.predict(X_all_s[hist_mask])

    # Infer trade part with warmup to avoid "new sequence" jump
    if trade_mask.any():
        first_trade_pos = int(np.where(trade_mask)[0][0])
        warm = int(max(0, infer_warmup_bars))
        warm_start_pos = max(0, first_trade_pos - warm)

        X_seg = X_all_s[warm_start_pos:]
        seg_states = best_model.predict(X_seg)

        offset = first_trade_pos - warm_start_pos
        states[trade_mask] = seg_states[offset:]

    return best_model, scaler, states, float(best_ll)


def run_hmm_stage_placeholder(argv=None) -> int:
    if argv and any(a in ("-h", "--help") for a in argv):
        print("usage: qrs-new-hmm [--help]")
        print("placeholder: hmm stage")
        return 0
    return 0
