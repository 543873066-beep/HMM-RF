import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def run_super_hmm_from_overlay(df_overlay: pd.DataFrame, cfg: "Config"):
    from quant_refactor_skeleton.features.feature_builder import pick_fit_mask_by_date
    from quant_refactor_skeleton.hmm.model import align_states_by_volatility
    from quant_refactor_skeleton.super_state.lifecycle import (
        compute_avg_run_local_expected_life_online_ewma,
        compute_run_len_so_far,
    )

    super_feature_cols = [
        "ret_1",
        "vol_short",
        "trend_strength",
        "adx_14",
        "rsi_14",
        "atr_14",
        "overlay_s5",
        "overlay_s30",
        "overlay_sd",
    ]
    df2 = df_overlay.dropna(subset=super_feature_cols).copy()
    if len(df2) == 0:
        raise ValueError("overlay after dropna is empty")

    X_all = df2[super_feature_cols].values.astype(float)

    # --- Determine infer-only split point (trade start) ---
    trade_start_dt = None
    if getattr(cfg, "trade_start_date", None):
        try:
            trade_start_dt = pd.to_datetime(cfg.trade_start_date)
        except Exception:
            trade_start_dt = None

    # Fit mask: by default uses cfg.hmm_train_*; if trade_start_dt is provided,
    # enforce "HMM classification cutoff before trade_df" (fit uses only < trade_start_dt).
    fit_mask = pick_fit_mask_by_date(
        df2.index,
        cfg.exclude_last_day_from_fit,
        train_start_date=cfg.hmm_train_start_date,
        train_end_date=cfg.hmm_train_end_date,
    )
    if trade_start_dt is not None:
        fit_mask = fit_mask & (df2.index < trade_start_dt)

    if fit_mask.sum() < max(300, 7 * 50):
        fit_mask[:] = True

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_all[fit_mask])

    best_k = 7
    best_model = None
    best_ll = -np.inf

    for i in range(int(cfg.num_inits)):
        current_seed = int(cfg.rs_super) + i
        try:
            model = GaussianHMM(
                n_components=best_k,
                covariance_type="full",
                n_iter=500,
                random_state=current_seed,
                verbose=False,
                min_covar=float(cfg.min_covar),
            )
            model.fit(X_fit_s)
        except ValueError:
            try:
                model = GaussianHMM(
                    n_components=best_k,
                    covariance_type="diag",
                    n_iter=500,
                    random_state=current_seed,
                    verbose=False,
                    min_covar=1e-2,
                )
                model.fit(X_fit_s)
            except Exception:
                continue

        ll = model.score(X_fit_s)
        if ll > best_ll:
            best_ll = ll
            best_model = model

    if best_model is None:
        best_model = GaussianHMM(
            n_components=best_k,
            covariance_type="diag",
            n_iter=500,
            random_state=int(cfg.rs_super),
            min_covar=0.1,
        )
        best_model.fit(X_fit_s)
        best_ll = best_model.score(X_fit_s)

    align_states_by_volatility(best_model)

    # stash fitted super HMM model for optional export
    try:
        cfg._super_model = best_model
        cfg._super_scaler = scaler
        cfg._super_feature_cols = list(super_feature_cols)
    except Exception:
        pass

    X_all_s = scaler.transform(X_all)
    idx = df2.index

    # --- Infer super_state with "infer-only on trade" ---
    if trade_start_dt is None:
        df2["super_state"] = best_model.predict(X_all_s)

        # NOTE: predict_proba is a smoother over the whole sequence.
        # In backtests it's fine; for strict real-time you would implement online filtering.
        try:
            gamma = best_model.predict_proba(X_all_s)
            df2["posterior_maxp"] = gamma.max(axis=1)
            ent = -np.sum(gamma * np.log(np.clip(gamma, 1e-12, 1.0)), axis=1)
            df2["posterior_entropy"] = ent
            df2["stability_score"] = 1.0 - (ent / np.log(best_k))
            df2["mixed_signals"] = 1.0 - df2["stability_score"]
        except Exception:
            pass
    else:
        hist_mask = idx < trade_start_dt
        trade_mask = ~hist_mask

        states = np.full(len(idx), -1, dtype=int)

        if hist_mask.any():
            states[hist_mask] = best_model.predict(X_all_s[hist_mask])

        if trade_mask.any():
            warm = int(getattr(cfg, "super_infer_warmup_bars", 200))
            first_trade_pos = int(np.where(trade_mask)[0][0])
            warm_start_pos = max(0, first_trade_pos - max(0, warm))
            X_seg = X_all_s[warm_start_pos:]
            seg_states = best_model.predict(X_seg)
            offset = first_trade_pos - warm_start_pos
            states[trade_mask] = seg_states[offset:]

        df2["super_state"] = states

        # Compute posterior features segment-wise (reduces "new-sequence" effect at trade_start).
        try:
            post_maxp = np.full(len(idx), np.nan, dtype=float)
            post_ent = np.full(len(idx), np.nan, dtype=float)

            if hist_mask.any():
                g_hist = best_model.predict_proba(X_all_s[hist_mask])
                post_maxp[hist_mask] = g_hist.max(axis=1)
                ent = -np.sum(g_hist * np.log(np.clip(g_hist, 1e-12, 1.0)), axis=1)
                post_ent[hist_mask] = ent

            if trade_mask.any():
                warm = int(getattr(cfg, "super_infer_warmup_bars", 200))
                first_trade_pos = int(np.where(trade_mask)[0][0])
                warm_start_pos = max(0, first_trade_pos - max(0, warm))
                X_seg = X_all_s[warm_start_pos:]
                g_seg = best_model.predict_proba(X_seg)
                offset = first_trade_pos - warm_start_pos
                g_trade = g_seg[offset:]
                post_maxp[trade_mask] = g_trade.max(axis=1)
                ent = -np.sum(g_trade * np.log(np.clip(g_trade, 1e-12, 1.0)), axis=1)
                post_ent[trade_mask] = ent

            df2["posterior_maxp"] = post_maxp
            df2["posterior_entropy"] = post_ent
            df2["stability_score"] = 1.0 - (post_ent / np.log(best_k))
            df2["mixed_signals"] = 1.0 - df2["stability_score"]
        except Exception:
            pass

    # Local expected lifetime and switch rate (for gate)  [causal, no future leak]
    s_arr = df2["super_state"].values.astype(int)
    win = int(getattr(cfg, "avg_run_window", 200))
    win = max(20, win)  # keep sane lower bound
    # Online expected-life + optional EWMA prior (matches live behavior)
    ewma_alpha = float(getattr(cfg, "ewma_alpha_run", 0.0))
    min_hist_runs = int(getattr(cfg, "min_hist_runs", 3))
    use_prior = bool(getattr(cfg, "use_ewma_prior", True))
    avg_arr, ewma_state = compute_avg_run_local_expected_life_online_ewma(
        s_arr,
        window=win,
        ewma_alpha=ewma_alpha,
        min_hist_runs=min_hist_runs,
        use_ewma_prior=use_prior,
    )
    df2["avg_run_local"] = avg_arr.astype(float)
    # --- NEW: current run length (age) + exhaustion ratio ---
    df2["run_len_so_far"] = compute_run_len_so_far(df2["super_state"].values)
    df2["exhaustion_ratio"] = df2["run_len_so_far"] / np.maximum(
        pd.to_numeric(df2["avg_run_local"], errors="coerce").astype(float).values,
        1e-6,
    )
    # stash ewma_state for export
    try:
        cfg._path_stats_ewma = ewma_state
    except Exception:
        pass

    # Switch rate: causal rolling mean of state changes (uses only past + current)
    df2["switch_rate_local"] = (df2["super_state"] != df2["super_state"].shift(1)).astype(float).rolling(
        50,
        min_periods=10,
    ).mean()

    return df2, best_k, float(best_ll)


def run_super_stage_placeholder(argv=None) -> int:
    if argv and any(a in ("-h", "--help") for a in argv):
        print("usage: qrs-new-super [--help]")
        print("placeholder: super-state stage")
        return 0
    return 0


def recompute_runlife_metrics(
    df: pd.DataFrame,
    avg_run_window: int = 200,
    ewma_alpha_run: float = 0.15,
    min_hist_runs: int = 3,
    use_ewma_prior: bool = True,
):
    from quant_refactor_skeleton.super_state.lifecycle import (
        compute_avg_run_local_expected_life_online_ewma,
        compute_run_len_so_far,
    )

    out = df.copy()
    if "super_state" not in out.columns:
        return out
    s_arr = pd.to_numeric(out["super_state"], errors="coerce").fillna(-1).astype(int).values
    avg_arr, _ = compute_avg_run_local_expected_life_online_ewma(
        s_arr,
        window=int(avg_run_window),
        ewma_alpha=float(ewma_alpha_run),
        min_hist_runs=int(min_hist_runs),
        use_ewma_prior=bool(use_ewma_prior),
    )
    out["avg_run_local"] = avg_arr.astype(float)
    out["run_len_so_far"] = compute_run_len_so_far(s_arr)
    out["exhaustion_ratio"] = out["run_len_so_far"] / np.maximum(
        pd.to_numeric(out["avg_run_local"], errors="coerce").astype(float).values,
        1e-6,
    )
    out["switch_rate_local"] = (pd.Series(s_arr, index=out.index) != pd.Series(s_arr, index=out.index).shift(1)).astype(
        float
    ).rolling(50, min_periods=10).mean()
    return out
