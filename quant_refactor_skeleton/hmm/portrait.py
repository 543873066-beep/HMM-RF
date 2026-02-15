import pandas as pd


def build_state_portrait(df_states: pd.DataFrame, state_col: str, tf_name: str) -> pd.DataFrame:
    g = df_states.groupby(state_col)
    prof = g.agg(
        count=("ret_1", "size"),
        ret_mean=("ret_1", "mean"),
        ret_std=("ret_1", "std"),
        vol_short_mean=("vol_short", "mean"),
        trend_strength_mean=("trend_strength", "mean"),
    ).reset_index()
    prof.insert(0, "tf", tf_name)
    return prof
