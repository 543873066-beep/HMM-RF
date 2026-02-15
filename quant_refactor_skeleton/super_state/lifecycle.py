import numpy as np


def compute_avg_run_local_expected_life(s_arr: np.ndarray, window: int) -> np.ndarray:
    """
    Causal "expected" lifetime for the *current* state based on recent completed runs.

    For each time t with state s_t:
      - maintain a deque of completed run lengths for each state, keyed by run end index
      - keep only runs whose end index is within the last `window` bars
      - avg_run_local[t] = mean(lengths of those recent completed runs for state s_t)
        If none exist yet, fall back to the current run length so far (>=1).

    This avoids future leak from "segment total length" and makes train/live consistent.
    """
    n = len(s_arr)
    out = np.full(n, np.nan, dtype=float)
    from collections import defaultdict, deque

    runs = defaultdict(deque)  # state -> deque[(end_idx, run_len)]

    prev_state = None
    cur_len = 0

    for t in range(n):
        s = int(s_arr[t])

        if prev_state is None:
            prev_state = s
            cur_len = 1
        else:
            if s == prev_state:
                cur_len += 1
            else:
                # finalize previous run ended at t-1
                runs[prev_state].append((t - 1, int(cur_len)))
                prev_state = s
                cur_len = 1

        dq = runs[s]
        cutoff = t - int(window)
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        if dq:
            out[t] = float(np.mean([rl for _, rl in dq]))
        else:
            out[t] = float(cur_len)

    return out


def compute_avg_run_local_expected_life_online_ewma(
    s_arr: np.ndarray,
    window: int,
    ewma_alpha: float = 0.15,
    min_hist_runs: int = 3,
    use_ewma_prior: bool = True,
):
    """
    Online + causal expected-life with optional EWMA prior fusion.

    For each t with current state s_t:
      - completed runs for each state are stored as (end_idx, run_len) within last `window`
      - mean_runs = mean(completed run lengths) for s_t (within window)
      - if n_runs >= min_hist_runs: avg_run_local = mean_runs
      - else:
          - if use_ewma_prior and ewma_alpha>0 and ewma_prior exists: blend toward ewma_prior
          - otherwise fallback to current run length so far

    EWMA prior is updated ONLY when a run completes (no future leak).
    Returns:
      avg_run_local: np.ndarray[float]
      ewma_state: dict[str, dict] (json-serializable) containing per-state ewma mean/n
    """
    from collections import defaultdict, deque

    s_arr = np.asarray(s_arr).astype(int)
    n = len(s_arr)
    out = np.full(n, np.nan, dtype=float)

    window = int(max(20, window))
    min_hist_runs = int(max(1, min_hist_runs))
    ewma_alpha = float(ewma_alpha)

    runs = defaultdict(deque)  # state -> deque[(end_idx, run_len)]
    sums = defaultdict(float)  # state -> sum(run_len in deque)
    ewma = {}  # state -> (mean, n)

    prev_state = None
    cur_len = 0

    def update_ewma(st: int, run_len: float):
        if not use_ewma_prior:
            return
        if ewma_alpha <= 0:
            return
        if st not in ewma:
            ewma[st] = (float(run_len), 1)
        else:
            m0, n0 = ewma[st]
            m1 = ewma_alpha * float(run_len) + (1.0 - ewma_alpha) * float(m0)
            ewma[st] = (float(m1), int(n0) + 1)

    def get_prior(st: int):
        if (not use_ewma_prior) or (ewma_alpha <= 0):
            return None
        if st in ewma and ewma[st][1] > 0 and ewma[st][0] > 0:
            return float(ewma[st][0])
        # optional global fallback (mean of existing priors)
        if ewma:
            ms = [m for (m, nn) in ewma.values() if nn > 0 and m > 0]
            if ms:
                return float(sum(ms) / len(ms))
        return None

    for t in range(n):
        s = int(s_arr[t])

        if prev_state is None:
            prev_state = s
            cur_len = 1
        else:
            if s == prev_state:
                cur_len += 1
            else:
                # finalize previous run ended at t-1
                runs[prev_state].append((t - 1, int(cur_len)))
                sums[prev_state] += float(cur_len)
                update_ewma(prev_state, float(cur_len))
                prev_state = s
                cur_len = 1

        # prune runs for current state by end_idx within window
        cutoff = t - window
        dq = runs[s]
        while dq and dq[0][0] < cutoff:
            end_i, L = dq.popleft()
            sums[s] -= float(L)
        if dq and sums[s] < 0:
            sums[s] = float(sum(L for _, L in dq))

        n_runs = len(dq)
        mean_runs = (sums[s] / float(n_runs)) if n_runs > 0 else None
        prior = get_prior(s)

        if n_runs >= min_hist_runs and (mean_runs is not None):
            avg = float(mean_runs)
        else:
            if (prior is None) or (prior <= 0):
                avg = float(cur_len)
            else:
                if mean_runs is None or n_runs <= 0:
                    avg = float(prior)
                else:
                    w = float(max(1, min_hist_runs - n_runs))
                    avg = (float(mean_runs) * float(n_runs) + float(prior) * w) / (float(n_runs) + w)

        out[t] = max(1.0, float(avg))

    ewma_state = {str(k): {"mean": float(v[0]), "n": int(v[1])} for k, v in ewma.items()}
    return out, ewma_state


def compute_run_len_so_far(s_arr: np.ndarray) -> np.ndarray:
    """Causal current run length for each time step (age of current state segment)."""
    s_arr = np.asarray(s_arr).astype(int)
    out = np.empty(len(s_arr), dtype=float)
    prev = None
    cur = 0
    for i, s in enumerate(s_arr):
        if prev is None or s != prev:
            cur = 1
        else:
            cur += 1
        out[i] = float(cur)
        prev = s
    return out


def compute_path_stats_seed_expected_life(
    s_arr: np.ndarray,
    window: int = 200,
    switch_window: int = 50,
    ewma_alpha: float = 0.15,
    min_hist_runs: int = 3,
    use_ewma_prior: bool = True,
):
    """Build a *causal* seed for path-dependent metrics used by gate (avg_run_local / switch_rate_local).

    - avg_run_local mode: expected-life (mean of *completed* run lengths in recent history)
    - optional EWMA prior: updated ONLY when a run completes (no future leak)
    - switch_rate_local: rolling mean of switches over the last `switch_window` bars

    Returns a JSON-serializable dict that live loader can warm-start from.
    """
    from collections import defaultdict, deque

    s_arr = np.asarray(s_arr).astype(int)
    n = len(s_arr)
    window = int(max(20, window))
    switch_window = int(max(5, switch_window))
    min_hist_runs = int(max(1, min_hist_runs))
    ewma_alpha = float(ewma_alpha)

    runs = defaultdict(deque)  # state -> deque[(end_idx, run_len)] pruned by end_idx within `window`
    sums = defaultdict(float)

    ewma = {}  # state -> (mean, n)

    prev_state = None
    cur_len = 0

    sw_buf = deque(maxlen=switch_window)

    def _update_ewma(st: int, run_len: float):
        if (not use_ewma_prior) or (ewma_alpha <= 0):
            return
        if st not in ewma:
            ewma[st] = (float(run_len), 1)
        else:
            m0, n0 = ewma[st]
            m1 = ewma_alpha * float(run_len) + (1.0 - ewma_alpha) * float(m0)
            ewma[st] = (float(m1), int(n0) + 1)

    for t in range(n):
        s = int(s_arr[t])

        if prev_state is None:
            prev_state = s
            cur_len = 1
            sw_buf.append(0.0)
        else:
            if s == prev_state:
                cur_len += 1
                sw_buf.append(0.0)
            else:
                # finalize previous run ended at t-1
                dq = runs[prev_state]
                dq.append((t - 1, int(cur_len)))
                sums[prev_state] += float(cur_len)
                _update_ewma(prev_state, float(cur_len))

                prev_state = s
                cur_len = 1
                sw_buf.append(1.0)

        # prune per-state runs for current state
        cutoff = t - window
        dq = runs[s]
        while dq and dq[0][0] < cutoff:
            end_i, L = dq.popleft()
            sums[s] -= float(L)
        if dq and sums[s] < 0:
            sums[s] = float(sum(L for _, L in dq))

    seed = {
        "path_metric_mode": "expected_life_online_ewma" if (use_ewma_prior and ewma_alpha > 0) else "expected_life",
        "avg_run_window": int(window),
        "switch_window": int(switch_window),
        "switch_min_periods": 10,
        "min_hist_runs": int(min_hist_runs),
        "use_ewma_prior": bool(use_ewma_prior),
        "ewma_alpha_run": float(ewma_alpha),
        "per_state_runs": {str(k): list(v) for k, v in runs.items()},
        "per_state_sums": {str(k): float(sums.get(k, 0.0)) for k in runs.keys()},
        "ewma": {str(k): {"mean": float(v[0]), "n": int(v[1])} for k, v in ewma.items()},
        "last_state": int(prev_state) if prev_state is not None else None,
        "cur_run_len": int(cur_len) if prev_state is not None else 0,
        "switch_buf": list(sw_buf),
        "seed_n": int(n),
    }
    return seed
