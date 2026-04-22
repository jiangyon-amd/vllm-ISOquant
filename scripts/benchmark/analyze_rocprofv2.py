#!/usr/bin/env python3
"""
分析 rocprofv2 产生的 kernel trace / HIP API trace CSV。

用法:
  # 分析单个目录（自动找 results_*.csv）
  python3 analyze_rocprofv2.py /tmp/rocprof_e2e_xxx
  python3 analyze_rocprofv2.py /data/jiangyon/vllm_rotation/moe_only_profile/fused

  # 分析单个 CSV
  python3 analyze_rocprofv2.py /path/to/results_1920607.csv

  # 按内核名过滤
  python3 analyze_rocprofv2.py /tmp/rocprof_e2e_xxx --filter "mfma_rot_quant,fused_rot"

  # 同时输出 HIP API 统计（若有 hip_api_trace_*.csv）
  python3 analyze_rocprofv2.py /path/to/dir --hip-api

  # 对比多份结果（如 hip vs gluon）
  python3 analyze_rocprofv2.py --compare /tmp/rocprof_e2e_xxx
"""
from __future__ import annotations

import argparse
import csv
import os
import glob
from collections import defaultdict
from pathlib import Path


def find_csvs(path: str):
    """若 path 是目录，找 results_*.csv 和 hip_api_trace_*.csv；若是文件则返回 [path]."""
    p = Path(path)
    if p.is_file():
        return [str(p)], []
    if not p.is_dir():
        return [], []
    results = sorted(glob.glob(str(p / "results*.csv")) + glob.glob(str(p / "results_*.csv")))
    results = list(dict.fromkeys(results))  # 去重保序
    hip = sorted(glob.glob(str(p / "hip_api_trace*.csv")))
    return results, hip


def shorten_kernel_name(name: str, max_len: int = 72) -> str:
    """缩短内核名：保留有意义部分（含 .kd 的符号名或最后 :: 段），便于汇总."""
    if len(name) <= max_len:
        return name
    # 仅含 .kd 的短名（如 _fused_rot_quant_v2.kd）直接返回
    if name.endswith(".kd") and len(name) <= max_len:
        return name
    # 去掉末尾 " (...) (.kd)"，得到模板部分
    if " (" in name:
        base = name.split(" (")[0].strip()
        if len(base) <= max_len:
            return base
        name = base
    # 取最后一个 :: 段（类/函数名）
    if "::" in name:
        segments = name.split("::")
        for i in range(len(segments) - 1, -1, -1):
            candidate = "::".join(segments[i:])
            if candidate and len(candidate) <= max_len:
                return candidate
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def parse_kernel_csv(csv_path: str) -> list[tuple[float, str, dict]]:
    """解析 kernel trace CSV，返回 [(duration_us, kernel_name, row_dict), ...]."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row["Start_Timestamp"])
                end = float(row["End_Timestamp"])
                name = row.get("Kernel_Name", "")
                # 时间戳一般为纳秒
                duration_us = (end - start) / 1000.0
                rows.append((duration_us, name, row))
            except (KeyError, ValueError):
                continue
    return rows


def parse_hip_api_csv(csv_path: str) -> list[tuple[float, str]]:
    """解析 HIP API trace CSV，返回 [(duration_us, api_name), ...]."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row["Start_Timestamp"])
                end = float(row["End_Timestamp"])
                name = row.get("Function", row.get("API", ""))
                duration_us = (end - start) / 1000.0
                rows.append((duration_us, name))
            except (KeyError, ValueError):
                continue
    return rows


def aggregate_kernels(rows: list[tuple[float, str, dict]]) -> dict[str, list[float]]:
    """按缩短后的内核名聚合 duration."""
    by_name: dict[str, list[float]] = defaultdict(list)
    for dur, name, _ in rows:
        short = shorten_kernel_name(name)
        by_name[short].append(dur)
    return dict(by_name)


def print_kernel_summary(
    agg: dict[str, list[float]],
    *,
    top_n: int = 20,
    filter_keywords: list[str] | None = None,
    title: str = "Kernel Summary",
):
    """打印内核汇总：总时间、调用次数、平均时间，以及 Top N."""
    if filter_keywords:
        agg = {
            k: v
            for k, v in agg.items()
            if any(kw in k for kw in filter_keywords)
        }
    if not agg:
        print(f"{title}: (no matching kernels)")
        return

    # 按总时间排序
    total_by_name = {k: sum(v) for k, v in agg.items()}
    count_by_name = {k: len(v) for k, v in agg.items()}
    sorted_names = sorted(total_by_name.keys(), key=lambda x: -total_by_name[x])

    print(f"\n{title}")
    print("=" * (len(title) + 2))
    print(f"{'Kernel (short)':<72} {'Count':>8} {'Total(µs)':>12} {'Avg(µs)':>10}")
    print("-" * 72, "-" * 8, "-" * 12, "-" * 10)
    for name in sorted_names[:top_n]:
        total = total_by_name[name]
        count = count_by_name[name]
        avg = total / count
        display_name = name[:70] + ".." if len(name) > 72 else name
        print(f"{display_name:<72} {count:>8} {total:>12.1f} {avg:>10.2f}")
    if len(sorted_names) > top_n:
        print(f"  ... and {len(sorted_names) - top_n} more kernel types")
    print()


def print_hip_api_summary(rows: list[tuple[float, str]], top_n: int = 25):
    """汇总 HIP API 调用时间."""
    by_name: dict[str, list[float]] = defaultdict(list)
    for dur, name in rows:
        by_name[name].append(dur)
    total_by_name = {k: sum(v) for k, v in by_name.items()}
    count_by_name = {k: len(v) for k, v in by_name.items()}
    sorted_names = sorted(total_by_name.keys(), key=lambda x: -total_by_name[x])
    print("\nHIP API Summary (by total time)")
    print("=" * 50)
    print(f"{'API':<45} {'Count':>8} {'Total(µs)':>12} {'Avg(µs)':>10}")
    print("-" * 45, "-" * 8, "-" * 12, "-" * 10)
    for name in sorted_names[:top_n]:
        total = total_by_name[name]
        count = count_by_name[name]
        avg = total / count
        display = name[:43] + ".." if len(name) > 45 else name
        print(f"{display:<45} {count:>8} {total:>12.1f} {avg:>10.2f}")
    print()


def run_compare(base_dir: str, filter_keywords: list[str] | None):
    """对比同一目录下多份 results_*.csv（如 results_hip.csv / results_gluon.csv）."""
    result_files = sorted(glob.glob(os.path.join(base_dir, "results*.csv")))
    result_files = [f for f in result_files if "hip_api" not in f.lower()]
    if len(result_files) < 2:
        print("Compare 需要目录下至少 2 个 results_*.csv 文件。")
        return
    labels = [Path(f).stem.replace("results_", "").replace("results", "").strip("_") or Path(f).name for f in result_files]
    all_rows = []
    for f in result_files:
        all_rows.append(parse_kernel_csv(f))
    # 统一按缩短名聚合，再按名对比
    all_agg = [aggregate_kernels(r) for r in all_rows]
    # 取所有出现过的内核名
    all_names = set()
    for agg in all_agg:
        all_names.update(agg.keys())
    if filter_keywords:
        all_names = {n for n in all_names if any(kw in n for kw in filter_keywords)}
    all_names = sorted(all_names, key=lambda x: -max((sum(all_agg[i].get(x, [])) for i in range(len(all_agg)))))
    print("\nCompare: Total time (µs) per kernel type")
    print("=" * 90)
    header = f"{'Kernel (short)':<52}"
    for lab in labels:
        header += f" {lab[:12]:>14}"
    print(header)
    print("-" * 52, *("-" * 14 for _ in labels))
    for name in all_names[:30]:
        display = name[:50] + ".." if len(name) > 52 else name
        line = f"{display:<52}"
        for agg in all_agg:
            total = sum(agg.get(name, []))
            line += f" {total:>14.1f}"
        print(line)
    print()


def main():
    ap = argparse.ArgumentParser(description="Analyze rocprofv2 kernel/HIP API trace CSV")
    ap.add_argument("path", nargs="?", default=".", help="目录或单个 results_*.csv 路径")
    ap.add_argument("-n", "--top", type=int, default=20, help="显示前 N 个内核 (default 20)")
    ap.add_argument("--filter", type=str, default="", help="逗号分隔的关键字，只显示名称包含这些的 (e.g. mfma_rot_quant,fused_rot)")
    ap.add_argument("--hip-api", action="store_true", help="同时解析并输出 hip_api_trace_*.csv 统计")
    ap.add_argument("--compare", action="store_true", help="对比目录下多份 results_*.csv")
    args = ap.parse_args()

    filter_keywords = [s.strip() for s in args.filter.split(",") if s.strip()]
    path = args.path

    if args.compare:
        if not os.path.isdir(path):
            print("--compare 需要指定目录")
            return
        run_compare(path, filter_keywords if filter_keywords else None)
        return

    result_csvs, hip_csvs = find_csvs(path)
    if not result_csvs:
        print("未找到 results_*.csv，请指定目录或 CSV 路径。")
        return

    for csv_path in result_csvs:
        label = Path(csv_path).stem
        rows = parse_kernel_csv(csv_path)
        if not rows:
            print(f"  {csv_path}: 无有效行")
            continue
        agg = aggregate_kernels(rows)
        total_all = sum(d for _, _, r in rows for d in [float(r.get("End_Timestamp", 0)) - float(r.get("Start_Timestamp", 0))] if r)
        total_all_us = sum(d for d, _, _ in rows)
        print(f"\nFile: {csv_path}")
        print(f"Total kernel dispatches: {len(rows)}, total time: {total_all_us:.1f} µs")
        print_kernel_summary(
            agg,
            top_n=args.top,
            filter_keywords=filter_keywords if filter_keywords else None,
            title=f"Kernel Summary — {label}",
        )

    if args.hip_api and hip_csvs:
        for hip_path in hip_csvs:
            rows = parse_hip_api_csv(hip_path)
            if rows:
                print(f"\nHIP API: {hip_path}")
                print_hip_api_summary(rows, top_n=args.top)


if __name__ == "__main__":
    main()
