# -*- coding: utf-8 -*-
"""
python rolling_runner.py

【进度条可视化 + 实验记录修复版】
1. 集成 tqdm 进度条：清晰展示 总体进度(Cycle)、Eval进度、Live进度。
2. 实验记录修复：修复夏普率计算时读取到时间列的Bug，确保 experiment_log.csv 正确生成。
3. 核心逻辑：保持原版 Robust Selector 不变。
"""

import os
import sys
import json
import subprocess
import shutil
import re
import datetime
import argparse
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp

# === 进度条库 ===
try:
    from tqdm import tqdm
except ImportError:
    print("请先安装 tqdm 库以显示进度条: pip install tqdm")
    def tqdm(iterable, **kwargs): return iterable

# ===================== 固定参数区 =====================
INPUT_CSV = r"data/sh000852_5m.csv"
ENGINE_PY = r"msp_engine_ewma_exhaustion_opt_atr_momo.py"
PYTHON_EXE = sys.executable

OUT_ROOT = r"outputs_roll"

# Live-pack exported by engine is placed under out_dir/<rf_subdir>/live_pack.
# Keep this in sync with the engine's cfg.rf_subdir (default in your engine).
RF_SUBDIR = "rf_h4_per_state_dynamic_selected"

START_ANCHOR_DATE = "2022-01-01"
FIRST_CUTOFF_MONTH = "2025-12"     # YYYY-MM
STEP_DAYS = 22
N_SWEEPS = 30

# ===================== 稳健 Top5 选择器 =====================
MIN_TRADES_EVAL = 8                 # 评估窗口内最少交易笔数
MIN_ACTIVE_BARS_EVAL = 60           # 评估窗口内最少 active bars
MIN_GATE_COVERAGE_EVAL = 0.03       # 评估窗口内 gate_on 覆盖率下限
MAX_GATE_COVERAGE_EVAL = 0.90       # 覆盖率上限
MAX_DD_EVAL = 0.08                  # 评估窗口内最大回撤上限（8%）

LAMBDA_DD = 3.0
LAMBDA_LOWTR = 0.15

EMA_ALPHA = 0.6                    # 新分数权重
ROBUST_FALLBACK_MODE = "relax"

DAY_END_HOUR = 15
DAY_START_HOUR = 9
DAY_START_MINUTE = 35

COMBO_MODE = "mean_valid"

EVAL_NJOBS = None
LIVE_NJOBS = None

# === 记录工具配置 ===
HISTORY_DIR = "history_records"
LOG_FILE = "experiment_log.csv"
# =================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def default_seed_triplet(i: int) -> Tuple[int, int, int]:
    return i * 13 + 5, i * 17 + 7, i * 19 + 11

def run_engine_once(run_dir, run_id, rs_5m, rs_30m, rs_1d, data_end_dt, trade_start_dt=None, trade_end_dt=None):
    ensure_dir(run_dir)
    cmd = [
        PYTHON_EXE, ENGINE_PY,
        "--input_csv", INPUT_CSV,
        "--input_tf_minutes", "5",
        "--out_dir", run_dir,
        "--enable_backtest", "1",
        "--run_id", run_id,
        "--rs_5m", str(rs_5m),
        "--rs_30m", str(rs_30m),
        "--rs_1d", str(rs_1d),
        "--data_end_date", str(pd.to_datetime(data_end_dt)),
        "--export_live_pack", "1",
        "--live_pack_dir", "live_pack",
    ]
    if trade_start_dt and trade_end_dt:
        cmd += ["--trade_start_date", str(pd.to_datetime(trade_start_dt)),
                "--trade_end_date", str(pd.to_datetime(trade_end_dt))]
    try:
        # 禁用子进程输出，避免打乱进度条
        with open(os.path.join(run_dir, "engine.log"), "w") as f_log:
            subprocess.check_call(cmd, stdout=f_log, stderr=f_log)
        return True, None
    except Exception as e:
        return False, repr(e)


def update_latest_live_pack(out_root: str, cycle_dir: str, cycle_id: str, top5: pd.DataFrame,
                            trade_start_dt: pd.Timestamp, trade_end_dt: pd.Timestamp) -> None:
    """Collect the latest artifacts needed for live trading.

    It overwrites out_root/live_pack_latest each cycle. Packs are copied from the LIVE run directories
    (because those correspond to hist-train + trade inference on the trade window).
    """
    try:
        latest_dir = os.path.join(out_root, "live_pack_latest")
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir, ignore_errors=True)
        os.makedirs(latest_dir, exist_ok=True)

        # save the last top5 seed selection (for direct live load)
        top5_path = os.path.join(latest_dir, "top5_seed_selection_latest.csv")
        top5.to_csv(top5_path, index=False, encoding="utf-8-sig")

        meta = {
            "cycle_id": cycle_id,
            "trade_start_dt": str(pd.to_datetime(trade_start_dt)),
            "trade_end_dt": str(pd.to_datetime(trade_end_dt)),
            "engine_py": ENGINE_PY,
            "input_csv": INPUT_CSV,
        }
        with open(os.path.join(latest_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # copy each selected LIVE pack
        for j in range(len(top5)):
            row = top5.iloc[j]
            rs5, rs30, rs1 = int(row.rs_5m), int(row.rs_30m), int(row.rs_1d)
            live_id = f"LIVE{j+1}_{rs5}_{rs30}_{rs1}"
            # Engine exports pack under: <run_dir>/<rf_subdir>/live_pack
            src_pack = os.path.join(cycle_dir, live_id, RF_SUBDIR, "live_pack")
            if os.path.exists(src_pack):
                dst = os.path.join(latest_dir, f"top{j+1}_{rs5}_{rs30}_{rs1}")
                shutil.copytree(src_pack, dst, dirs_exist_ok=True)
    except Exception:
        pass

def load_run_summary_optional(run_dir: str) -> Optional[Dict]:
    for root, _, files in os.walk(run_dir):
        if "run_summary.json" in files:
            try:
                with open(os.path.join(root, "run_summary.json"), "r", encoding="utf-8") as f:
                    return json.load(f)
            except: return None
    return None

def find_equity_curve_csv_optional(run_dir: str) -> Optional[str]:
    for root, _, files in os.walk(run_dir):
        if "backtest_equity_curve.csv" in files:
            return os.path.join(root, "backtest_equity_curve.csv")
    return None

def read_equity_curve(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = "time" if "time" in df.columns else df.columns[0]
    eq_col = None
    for c in ["equity", "eq", "equity_net", "equity_curve", "eq_net"]:
        if c in df.columns:
            eq_col = c
            break
    if eq_col is None:
        eq_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    return df[[tcol, eq_col]].rename(columns={tcol: "time", eq_col: "equity"})

def rescale_equity_to_1(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    base = float(df["equity"].iloc[0])
    if base == 0: return df
    out = df.copy()
    out["equity"] = out["equity"] / base
    return out

def make_flat_curve_from_input(trade_start_dt: pd.Timestamp, trade_end_dt: pd.Timestamp, value: float) -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, usecols=["time"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[(df["time"] >= trade_start_dt) & (df["time"] <= trade_end_dt)].sort_values("time")
    if df.empty:
        t = pd.to_datetime([trade_start_dt, trade_end_dt])
        return pd.DataFrame({"time": t, "equity": [value, value]})
    return pd.DataFrame({"time": df["time"].values, "equity": value})

def safe_mean_combo(df: pd.DataFrame, cols: List[str], mode: str) -> pd.Series:
    if mode == "mean_fixed5": return df[cols].mean(axis=1)
    return df[cols].mean(axis=1, skipna=True)

def resolve_njobs() -> Tuple[int, int]:
    cpu = mp.cpu_count()
    eval_n = EVAL_NJOBS if EVAL_NJOBS is not None else max(1, cpu - 1)
    live_n = LIVE_NJOBS if LIVE_NJOBS is not None else min(5, eval_n)
    live_n = max(1, live_n)
    return int(eval_n), int(live_n)

# ... [保留原有的 _safe_get, extract_eval_metrics, robust_select_top5 等函数，完全不变] ...
def _safe_get(d: dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def extract_eval_metrics_from_summary(summ: Optional[dict]) -> Dict:
    if not isinstance(summ, dict):
        return {"eval_sharpe_daily": float("-inf"), "eval_max_dd": float("nan"), "eval_trades": 0, "eval_active_bars": 0, "eval_gate_coverage": float("nan")}
    sharpe = float(_safe_get(summ, ["rf_stats", "sharpe_daily"], float("-inf")))
    max_dd = _safe_get(summ, ["rf_stats", "max_drawdown_daily"], None) or _safe_get(summ, ["rf_stats", "max_dd_daily"], None) or _safe_get(summ, ["rf_stats", "max_drawdown"], None)
    max_dd = float(max_dd) if max_dd is not None else float("nan")
    n_trades = _safe_get(summ, ["rf_stats", "n_trades"], None) or _safe_get(summ, ["rf_stats", "trades"], None)
    n_trades = int(n_trades) if n_trades is not None else 0
    active = _safe_get(summ, ["rf_stats", "active_bars"], None) or _safe_get(summ, ["rf_stats", "gated_n"], None) or _safe_get(summ, ["rf_stats", "gated_n_trade"], None)
    active = int(active) if active is not None else 0
    gate_cov = _safe_get(summ, ["rf_stats", "gate_coverage"], None) or _safe_get(summ, ["rf_stats", "gate_coverage_trade"], None) or _safe_get(summ, ["gate", "coverage"], None)
    gate_cov = float(gate_cov) if gate_cov is not None else float("nan")
    return {"eval_sharpe_daily": sharpe, "eval_max_dd": max_dd, "eval_trades": n_trades, "eval_active_bars": active, "eval_gate_coverage": gate_cov}

def _json_sanitize(obj):
    if isinstance(obj, dict): return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if (not np.isfinite(obj)) or np.isnan(obj): return None
        return float(obj)
    return obj

def robust_select_top5(
    eval_df: pd.DataFrame,
    seed_hist: Dict[str, Dict],
    mode: str = "relax"
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    df = eval_df.copy()

    for c in ["eval_trades","eval_active_bars","eval_gate_coverage","eval_max_dd","eval_sharpe_daily"]:
        if c not in df.columns:
            df[c] = np.nan

    # 修改：将过滤逻辑拆分
    def apply_filters(d: pd.DataFrame, strict_dd: bool = True, strict_trades: bool = True) -> pd.DataFrame:
        out = d[d["engine_ok"] == True].copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        # 必须有夏普率
        out = out[pd.to_numeric(out["eval_sharpe_daily"], errors="coerce").notna()]
        
        # 1. 交易次数限制 (通常坚守)
        if strict_trades:
            out = out[out["eval_trades"].fillna(0).astype(int) >= MIN_TRADES_EVAL]
            out = out[out["eval_active_bars"].fillna(0).astype(int) >= MIN_ACTIVE_BARS_EVAL]
            
            # Gate Coverage 如果有数据也应该检查
            gc = pd.to_numeric(out["eval_gate_coverage"], errors="coerce")
            # 如果全是 NaN (旧引擎)，则不筛；如果有值，则筛
            if gc.notna().sum() > 0:
                mask_gc = gc.isna() | ((gc >= MIN_GATE_COVERAGE_EVAL) & (gc <= MAX_GATE_COVERAGE_EVAL))
                out = out[mask_gc]

        # 2. 回撤限制 (可以放宽)
        if strict_dd:
            mdd = pd.to_numeric(out["eval_max_dd"], errors="coerce")
            mdd = mdd.replace([np.inf, -np.inf], np.nan).fillna(MAX_DD_EVAL * 2.0)
            mdd = mdd.clip(lower=0.0)
            out = out[mdd <= MAX_DD_EVAL]
            
        return out

    def add_score(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        sharpe = pd.to_numeric(out["eval_sharpe_daily"], errors="coerce").fillna(-np.inf)
        mdd = pd.to_numeric(out["eval_max_dd"], errors="coerce")
        mdd = mdd.replace([np.inf, -np.inf], np.nan).fillna(MAX_DD_EVAL * 2.0)
        mdd = mdd.clip(lower=0.0)
        trades = out["eval_trades"].fillna(0).astype(int)
        
        # trades penalty
        lowtr_pen = (np.maximum(0, MIN_TRADES_EVAL - trades)) * LAMBDA_LOWTR
        
        out["robust_score"] = sharpe - LAMBDA_DD * mdd - lowtr_pen
        
        rs = pd.to_numeric(out["robust_score"], errors="coerce")
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(-1e9)
        out["robust_score"] = rs
        return out

    # === 筛选逻辑升级 ===
    
    # 第1轮：全严格筛选
    cand = apply_filters(df, strict_dd=True, strict_trades=True)
    
    # 第2轮：如果不足，尝试放宽回撤
    if len(cand) < 5 and mode == "relax":
        cand_relaxed = apply_filters(df, strict_dd=False, strict_trades=True)
        
        # 【修改方案 A：宁缺毋滥】
        # 只要 Round 2 有选出东西（哪怕只有 1 个），就用 Round 2 的结果
        # 不再进入 Round 3 兜底
        if len(cand_relaxed) > 0:
            cand = cand_relaxed
        # 如果连 Round 2 都没选出任何东西，才考虑 Round 3 或者报错
        else:
            cand_final = df[(df["engine_ok"] == True) & (df["eval_trades"] > 0)].copy()
            if len(cand_final) > 0:
                cand = cand_final

    # 注意：这样修改后，返回的 top5 可能不足 5 行（比如只有 2 行）
    # 后续代码（update_latest_live_pack）是支持 len(top5) < 5 的，会自动适配

    # 计算分数
    cand = add_score(cand)
    cand["seed_triplet"] = cand["rs_5m"].astype(int).astype(str) + "_" + cand["rs_30m"].astype(int).astype(str) + "_" + cand["rs_1d"].astype(int).astype(str)

    # EMA 更新
    updated = dict(seed_hist)
    ema_scores = []
    for _, r in cand.iterrows():
        sid = r["seed_triplet"]
        score = float(r["robust_score"])
        old = updated.get(sid, {}).get("ema_score", None)
        ema = score if old is None or (not np.isfinite(old)) else (EMA_ALPHA * score + (1.0 - EMA_ALPHA) * float(old))
        updated.setdefault(sid, {})
        updated[sid]["ema_score"] = float(ema)
        updated[sid]["last_score"] = float(score)
        updated[sid]["last_sharpe"] = float(r["eval_sharpe_daily"])
        ema_scores.append(ema)

    cand["ema_score"] = ema_scores
    
    # 排序
    cand = cand.sort_values(["ema_score", "robust_score", "eval_sharpe_daily"], ascending=False).reset_index(drop=True)

    top5 = cand.head(5).copy()
    return top5, updated

def read_trading_days_5m(csv_path: str, start_date: str) -> List[pd.Timestamp]:
    df = pd.read_csv(csv_path, usecols=["time"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[df["time"] >= pd.to_datetime(start_date)]
    days = sorted(pd.Series(df["time"].dt.normalize().unique()).tolist())
    return days

def pick_backstep_endpoints(days: List[pd.Timestamp], step: int, min_start: pd.Timestamp) -> List[pd.Timestamp]:
    endpoints = []
    idx = len(days) - 1
    while idx >= 0:
        d = days[idx]
        if d < min_start: break
        endpoints.append(d)
        idx -= step
    return endpoints

def find_first_cutoff_in_month(endpoints: List[pd.Timestamp], ym: str) -> pd.Timestamp:
    y, m = map(int, ym.split("-"))
    cands = [d for d in endpoints if d.year == y and d.month == m]
    if not cands: raise ValueError(f"找不到落在 {ym} 的 22 交易日端点")
    return max(cands)

# ============================================================
# 实验记录模块 (Auto Recorder) [修正版]
# ============================================================
def parse_engine_cost():
    """从 ENGINE_PY 文件中提取 cost_bps_per_unit"""
    cost_val = "N/A"
    if os.path.exists(ENGINE_PY):
        with open(ENGINE_PY, 'r', encoding='utf-8') as f:
            content = f.read()
            m = re.search(r'cost_bps_per_unit.*?=\s*([\d\.]+)', content)
            if m: cost_val = m.group(1)
    return cost_val

def record_experiment_log(equity_csv_path: str, tag: str = "AutoRun"):
    """计算指标并写入日志"""
    if not os.path.exists(equity_csv_path): return

    try:
        df = pd.read_csv(equity_csv_path)
        if df.empty: return
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        # 1. 净值
        start_eq = df.iloc[0]['eq_global']
        end_eq = df.iloc[-1]['eq_global']
        final_nv = end_eq / start_eq
        
        # 2. 夏普 (修复: 明确使用 eq_global 列)
        df['date'] = df['time'].dt.date
        # 关键修正：显式指定列，防止 groupby 混入 time 列
        daily = df.groupby('date')['eq_global'].last()
        daily_ret = daily.pct_change().fillna(0)
        
        if daily_ret.std() == 0:
            sharpe = 0.0
        else:
            sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std()
            
    except Exception as e:
        print(f"[Recorder] Metrics calc failed: {e}")
        final_nv = 0.0
        sharpe = 0.0

    cost = parse_engine_cost()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 按照您的要求，记录所有关键参数
    record = {
        "Timestamp": timestamp,
        "Tag": tag,
        "STEP_DAYS": STEP_DAYS,
        "N_SWEEPS": N_SWEEPS,
        "FIRST_CUTOFF_MONTH": FIRST_CUTOFF_MONTH,
        "EMA_ALPHA": EMA_ALPHA,
        "COMBO_MODE": COMBO_MODE,
        "cost_bps": cost,
        "Final_Net_Value": round(final_nv, 4),
        "Sharpe_Ratio": round(sharpe, 4),
        "Run_ID": run_id
    }
    
    df_new = pd.DataFrame([record])
    if os.path.exists(LOG_FILE):
        df_new.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df_new.to_csv(LOG_FILE, mode='w', header=True, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 实验已记录: Return={record['Final_Net_Value']}, Sharpe={record['Sharpe_Ratio']}")
    
    # 归档
    os.makedirs(HISTORY_DIR, exist_ok=True)
    shutil.copy(equity_csv_path, os.path.join(HISTORY_DIR, f"equity_{run_id}_{tag}.csv"))
    if os.path.exists(os.path.join(OUT_ROOT, "monthly_top5_selections.csv")):
        shutil.copy(os.path.join(OUT_ROOT, "monthly_top5_selections.csv"), os.path.join(HISTORY_DIR, f"select_{run_id}_{tag}.csv"))

# ============================================================
# Main Loop (With TQDM)
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="AutoRun", help="Tag for experiment")
    args, unknown = parser.parse_known_args()

    eval_njobs, live_njobs = resolve_njobs()
    print(f"[PARALLEL] eval_njobs={eval_njobs} live_njobs={live_njobs} (cpu={mp.cpu_count()})")

    ensure_dir(OUT_ROOT)
    runs_root = os.path.join(OUT_ROOT, "runs")
    ensure_dir(runs_root)
    
    # === [新增] 强制清除历史记录，保证本次回测纯净 ===
    hist_json_path = os.path.join(OUT_ROOT, "seed_score_history.json")
    if os.path.exists(hist_json_path):
        print(f"[Init] 检测到旧的历史记录 {hist_json_path}，正在删除以确保回测纯净...")
        os.remove(hist_json_path)
    # =================================================
    
    print(f"Loading dates from {INPUT_CSV}...")
    days = read_trading_days_5m(INPUT_CSV, START_ANCHOR_DATE)
    if len(days) < STEP_DAYS * 2: raise ValueError("交易日太少")
    
    endpoints = pick_backstep_endpoints(days, STEP_DAYS, pd.to_datetime(START_ANCHOR_DATE))
    cutoff0 = find_first_cutoff_in_month(endpoints, FIRST_CUTOFF_MONTH)
    
    day_idx = {d: i for i, d in enumerate(days)}
    start_i = day_idx[cutoff0]
    end_i = len(days) - STEP_DAYS - 1

    # 生成所有 Cycle 索引
    cycle_indices = list(range(start_i, end_i + 1, STEP_DAYS))
    print(f"[INFO] Total Cycles to Run: {len(cycle_indices)}")

    global_eq = 1.0
    global_curves = []
    selections_all = []
    debug_rows = []

    # --- 总体进度条 ---
    pbar_total = tqdm(cycle_indices, desc="总体进度", unit="cycle")

    for k, ci in enumerate(pbar_total):
        cutoff_day = days[ci]
        next_start = days[ci + 1]
        next_end = days[ci + STEP_DAYS]

        cutoff_end_dt = pd.Timestamp(cutoff_day.date()) + pd.Timedelta(hours=DAY_END_HOUR)
        trade_start_dt = pd.Timestamp(next_start.date()) + pd.Timedelta(hours=DAY_START_HOUR, minutes=DAY_START_MINUTE)
        trade_end_dt = pd.Timestamp(next_end.date()) + pd.Timedelta(hours=DAY_END_HOUR)
        eval_start_idx = max(0, ci - STEP_DAYS + 1)
        eval_start_day = days[eval_start_idx]
        eval_start_dt = pd.Timestamp(eval_start_day.date()) + pd.Timedelta(hours=DAY_START_HOUR, minutes=DAY_START_MINUTE)

        cycle_id = f"C{k:03d}_{cutoff_day.date()}_to_{next_end.date()}"
        cycle_dir = os.path.join(runs_root, cycle_id)
        ensure_dir(cycle_dir)

        # 使用 tqdm.write 打印，避免打断进度条
        tqdm.write(f"\n>>> Cycle {k+1}/{len(cycle_indices)}: {cutoff_day.date()} -> Trade: {next_start.date()}~{next_end.date()}")

        cycle_debug = {"cycle_id": cycle_id, "live_ok": [], "live_missing": []}

        # A) Eval (带子进度条)
        def eval_one(i):
            rs5, rs30, rs1 = default_seed_triplet(i)
            run_id = f"S{i:03d}"
            run_dir = os.path.join(cycle_dir, run_id)
            ok, err = run_engine_once(run_dir, run_id, rs5, rs30, rs1, cutoff_end_dt, eval_start_dt, cutoff_end_dt)
            summ = load_run_summary_optional(run_dir) if ok else None
            metrics = extract_eval_metrics_from_summary(summ)
            sharpe = float("-inf")
            if summ and "rf_stats" in summ: sharpe = float(summ["rf_stats"].get("sharpe_daily", -999))
            return {"sweep_i": i, "run_id": run_id, "rs_5m": rs5, "rs_30m": rs30, "rs_1d": rs1,
                    "eval_sharpe_daily": float(metrics.get("eval_sharpe_daily", sharpe)),
                    "eval_max_dd": float(metrics.get("eval_max_dd", np.nan)),
                    "eval_trades": int(metrics.get("eval_trades", 0)),
                    "eval_active_bars": int(metrics.get("eval_active_bars", 0)),
                    "eval_gate_coverage": float(metrics.get("eval_gate_coverage", np.nan)),
                    "engine_ok": bool(ok), "engine_err": err}

        # tqdm 包装 Parallel 迭代器
        eval_rows = Parallel(n_jobs=eval_njobs, verbose=0)(
            delayed(eval_one)(i) for i in tqdm(range(N_SWEEPS), desc="  Eval", leave=False)
        )
        
        eval_df = pd.DataFrame(eval_rows).sort_values("eval_sharpe_daily", ascending=False).reset_index(drop=True)
        eval_df.to_csv(os.path.join(cycle_dir, "eval_all.csv"), index=False, encoding="utf-8-sig")

        # 稳健选择
        if not os.path.exists(os.path.join(OUT_ROOT, "seed_score_history.json")): seed_hist = {}
        else: 
            try: seed_hist = json.load(open(os.path.join(OUT_ROOT, "seed_score_history.json"), "r", encoding="utf-8"))
            except: seed_hist = {}

        top5, seed_hist = robust_select_top5(eval_df, seed_hist, mode=ROBUST_FALLBACK_MODE)
        try: json.dump(_json_sanitize(seed_hist), open(os.path.join(OUT_ROOT, "seed_score_history.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except: pass
        
        top5.to_csv(os.path.join(cycle_dir, "top5_selection.csv"), index=False, encoding="utf-8-sig")
        tqdm.write(f"  [Select] Top1 Sharpe: {top5['eval_sharpe_daily'].iloc[0]:.4f}" if not top5.empty else "  [Select] None")

        # B) Live (带子进度条)
        def live_one(j, row):
            rs5, rs30, rs1 = int(row.rs_5m), int(row.rs_30m), int(row.rs_1d)
            live_id = f"LIVE{j+1}_{rs5}_{rs30}_{rs1}"
            run_dir = os.path.join(cycle_dir, live_id)
            ok, err = run_engine_once(run_dir, live_id, rs5, rs30, rs1, trade_end_dt, trade_start_dt, trade_end_dt)
            eq_path = find_equity_curve_csv_optional(run_dir) if ok else None
            if not eq_path or not os.path.exists(eq_path): return {"ok": False, "live_id": live_id}
            try:
                eq = read_equity_curve(eq_path)
                eq = eq[(eq["time"] >= trade_start_dt) & (eq["time"] <= trade_end_dt)].copy()
                if eq.empty: return {"ok": False, "live_id": live_id}
                eq_rel = rescale_equity_to_1(eq).rename(columns={"equity": f"eq_rel_{j+1}"})
                return {"ok": True, "col": f"eq_rel_{j+1}", "eq_rel": eq_rel, "live_id": live_id, "eq_path": eq_path}
            except: return {"ok": False, "live_id": live_id}

        live_inputs = [(j, top5.iloc[j]) for j in range(len(top5))]
        live_results = Parallel(n_jobs=live_njobs, verbose=0)(
            delayed(live_one)(j, r) for (j, r) in tqdm(live_inputs, desc="  Live", leave=False)
        )

        live_curves = []
        for res in live_results:
            if res.get("ok"):
                live_curves.append((res["col"], res["eq_rel"]))
                cycle_debug["live_ok"].append({"live_id": res["live_id"], "eq_path": res["eq_path"]})
            else:
                cycle_debug["live_missing"].append(res)

        if not live_curves:
            flat = make_flat_curve_from_input(trade_start_dt, trade_end_dt, 1.0)
            merged = flat.rename(columns={"equity": "eq_rel_combo"})
            merged["eq_global"] = merged["eq_rel_combo"] * global_eq
            tqdm.write("  [Live] 0 valid runs -> Flat curve")
            debug_rows.append({"cycle_id": cycle_id, "end_global_eq": global_eq, "valid": 0})
            global_curves.append(merged[["time", "eq_global"]].rename(columns={"eq_global": "eq_global"}))
        else:
            merged = None
            for name, df in live_curves:
                merged = df if merged is None else merged.merge(df, on="time", how="outer")
            merged = merged.sort_values("time").ffill().fillna(1.0)
            rel_cols = [c for c in merged.columns if c.startswith("eq_rel_")]
            if COMBO_MODE == "mean_fixed5":
                while len(rel_cols) < 5:
                    merged[f"pad_{len(rel_cols)}"] = 1.0; rel_cols.append(f"pad_{len(rel_cols)}")
            merged["eq_rel_combo"] = safe_mean_combo(merged, rel_cols, COMBO_MODE)
            merged["eq_global"] = merged["eq_rel_combo"] * global_eq
            global_eq = float(merged["eq_global"].iloc[-1])
            merged.to_csv(os.path.join(cycle_dir, "live_equity_curves.csv"), index=False, encoding="utf-8-sig")
            tqdm.write(f"  [Live] Valid: {len(live_curves)} | New Eq: {global_eq:.4f}")
            debug_rows.append({"cycle_id": cycle_id, "end_global_eq": global_eq, "valid": len(live_curves)})
            global_curves.append(merged[["time", "eq_global"]].rename(columns={"eq_global": "eq_global"}))

        sel = top5.copy()
        sel["cycle_id"] = cycle_id
        sel["trade_start"] = str(trade_start_dt)
        sel["trade_end"] = str(trade_end_dt)
        selections_all.append(sel)
        json.dump(cycle_debug, open(os.path.join(cycle_dir, "cycle_debug.json"), "w"), indent=2)

        # overwrite latest live pack (models + params + trade snapshot) for direct live reading
        if isinstance(top5, pd.DataFrame) and (not top5.empty):
            update_latest_live_pack(OUT_ROOT, cycle_dir, cycle_id, top5, trade_start_dt, trade_end_dt)

    # 汇总
    print("\nStitching global curve...")
    all_global = pd.concat(global_curves, ignore_index=True).sort_values("time").drop_duplicates("time", keep="last")
    equity_csv_path = os.path.join(OUT_ROOT, "equity_global.csv")
    all_global.to_csv(equity_csv_path, index=False, encoding="utf-8-sig")

    if selections_all:
        pd.concat(selections_all, ignore_index=True).to_csv(os.path.join(OUT_ROOT, "monthly_top5_selections.csv"), index=False, encoding="utf-8-sig")
    
    pd.DataFrame(debug_rows).to_csv(os.path.join(OUT_ROOT, "roll_debug_summary.csv"), index=False, encoding="utf-8-sig")
    
    # 自动记录
    record_experiment_log(equity_csv_path, args.tag)

if __name__ == "__main__":
    main()