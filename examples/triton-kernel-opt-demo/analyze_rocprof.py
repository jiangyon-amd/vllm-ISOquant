import argparse
import csv
import json
import re
from pathlib import Path


NAME_KEYS = [
    "kernel_name",
    "kernel",
    "name",
]

DURATION_KEYS = [
    "duration_us",
    "duration_usec",
    "duration_usecs",
    "average_us",
    "time_us",
    "avg_us",
    "duration",
    "average",
    "avg",
]

START_KEYS = [
    "start_timestamp",
    "start",
    "begin_timestamp",
]

END_KEYS = [
    "end_timestamp",
    "end",
    "complete_timestamp",
]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def parse_number(value: str):
    value = value.strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def row_value(row, candidates):
    normalized = {normalize(key): value for key, value in row.items()}
    for key in candidates:
        if key in normalized and normalized[key] is not None:
            return normalized[key]
    return None


def duration_from_row(row):
    normalized = {normalize(key): value for key, value in row.items()}
    for key in DURATION_KEYS:
        if key in normalized:
            number = parse_number(normalized[key])
            if number is None:
                continue
            if key.endswith("_us") or "usec" in key or "time_us" == key:
                return number
            return number

    start_value = row_value(row, START_KEYS)
    end_value = row_value(row, END_KEYS)
    if start_value is None or end_value is None:
        return None

    start = parse_number(start_value)
    end = parse_number(end_value)
    if start is None or end is None:
        return None
    return (end - start) / 1000.0


def collect_records(input_dir: Path):
    records = []
    csv_files = sorted(input_dir.rglob("*.csv"))
    for csv_path in csv_files:
        with csv_path.open() as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for row in reader:
                name = row_value(row, NAME_KEYS)
                if name is None:
                    continue
                duration_us = duration_from_row(row)
                if duration_us is None or duration_us <= 0:
                    continue
                records.append(
                    {
                        "file": str(csv_path),
                        "kernel_name": name,
                        "duration_us": duration_us,
                    }
                )
    return records


def aggregate_records(records):
    aggregated = {}
    for record in records:
        key = record["kernel_name"]
        entry = aggregated.setdefault(
            key,
            {
                "kernel_name": key,
                "count": 0,
                "total_us": 0.0,
                "max_us": 0.0,
            },
        )
        entry["count"] += 1
        entry["total_us"] += record["duration_us"]
        entry["max_us"] = max(entry["max_us"], record["duration_us"])
    return sorted(aggregated.values(), key=lambda item: item["total_us"], reverse=True)


def choose_target(aggregated, variant: str):
    hints = {
        "baseline": ["baseline_matmul_kernel", "baseline", "matmul"],
        "round2": ["round2_matmul_kernel", "round2", "matmul"],
        "round3": ["round3_matmul_kernel", "round3", "matmul"],
    }.get(variant, ["matmul"])
    for hint in hints:
        for entry in aggregated:
            if hint in entry["kernel_name"].lower():
                return entry
    return aggregated[0] if aggregated else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--variant", choices=["baseline", "round2", "round3"], required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")

    records = collect_records(args.input)
    if not records:
        raise RuntimeError(f"No kernel-duration records found under: {args.input}")

    aggregated = aggregate_records(records)
    target = choose_target(aggregated, args.variant)

    summary = {
        "variant": args.variant,
        "input_dir": str(args.input),
        "total_records": len(records),
        "top_kernels": aggregated[:5],
        "target_kernel": target,
    }

    print(f"variant={args.variant}")
    print(f"input_dir={args.input}")
    print(f"records={len(records)}")
    if target is not None:
        print(f"target_kernel={target['kernel_name']}")
        print(f"target_total_us={target['total_us']:.3f}")
        print(f"target_count={target['count']}")

    print("top_kernels_by_total_us:")
    for entry in aggregated[:5]:
        print(
            f"  total_us={entry['total_us']:.3f} "
            f"count={entry['count']} "
            f"max_us={entry['max_us']:.3f} "
            f"name={entry['kernel_name']}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
