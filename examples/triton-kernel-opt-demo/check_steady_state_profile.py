import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-target-dispatches", type=int, required=True)
    args = parser.parse_args()

    data = json.loads(args.summary_json.read_text())
    target = data.get("target_kernel")
    if not target:
        raise RuntimeError("No target kernel found in the summary.")

    count = int(target["count"])
    name = target["kernel_name"]
    print(f"target_kernel={name}")
    print(f"target_dispatch_count={count}")
    print(f"max_allowed={args.max_target_dispatches}")

    if count > args.max_target_dispatches:
        raise SystemExit(
            f"Profile is not steady-state: target dispatch count {count} "
            f"exceeds limit {args.max_target_dispatches}."
        )

    print("steady_state_profile=pass")


if __name__ == "__main__":
    main()
