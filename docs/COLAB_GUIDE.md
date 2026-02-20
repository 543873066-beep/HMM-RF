# Colab 使用指引

适用场景：在 Google Colab（Linux）里运行 new route 的 compare / stage-diff / rolling，对齐验证不依赖 PowerShell。

## 1. 准备工作
在 Colab 中执行：
```bash
!git clone <你的仓库地址>
%cd HMM-RF
```

安装依赖：
```bash
!pip install -r requirements.txt
```

## 2. 放入数据
你需要一份 5m OHLCV CSV，至少包含列：
`time, open, high, low, close, volume`

可上传到 `data/`：
```bash
# 在 Colab 左侧文件面板上传到 data/ 目录
!ls data
```

或者直接指定路径：
```bash
# 假设你上传到了 /content/my_data/sh000852_5m.csv
```

## 3. 最常用的 3 条命令

### 3.1 Engine compare（最重要）
```bash
!python scripts/run_qrs.py --mode compare --route new --input_csv data/sh000852_5m.csv --disable_legacy_fallback
```
期望输出：
```
compare=PASS
rows_over_threshold=0
```

### 3.2 Stage diff（查找差异来源）
```bash
!python scripts/run_qrs.py --mode stage-diff --input_csv data/sh000852_5m.csv --out_root outputs_rebuild/qrs_runs
```
期望输出：features/super_state/rf_inputs 的 diff 在噪声级。

### 3.3 Rolling（对比 fold）
```bash
!python scripts/run_qrs.py --mode rolling --route new --input_csv data/sh000852_5m.csv --disable_legacy_fallback
```
rolling 可能比较慢，属于正常现象。

## 4. 常见问题
1) **找不到 CSV**
   - 确认数据在 `data/` 或用 `--input_csv` 指定绝对路径  
2) **运行很慢**
   - HMM/RF + rolling 本身耗时  
3) **HMM warning**
   - “Model is not converging” 是训练告警，不等于失败  
4) **输出在哪**
   - 默认 `outputs_rebuild/qrs_runs/<timestamp>/...`

## 5. 最小回归检查清单
1) compare 输出 `compare=PASS`  
2) `rows_over_threshold=0`  
3) stage-diff 关键字段差异接近 0  

