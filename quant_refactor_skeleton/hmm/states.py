import numpy as np
import pandas as pd


def transition_matrix_from_states(states: np.ndarray, n_states: int) -> pd.DataFrame:
    mat = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        mat[int(a), int(b)] += 1.0
    row_sum = mat.sum(axis=1, keepdims=True)
    prob = mat / np.where(row_sum == 0, 1.0, row_sum)
    df = pd.DataFrame(prob, columns=[f"to_{i}" for i in range(n_states)])
    df.insert(0, "from_state", [f"{i}" for i in range(n_states)])
    return df


def mode_agg(x: pd.Series):
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    vc = x.value_counts()
    return vc.index[0]
