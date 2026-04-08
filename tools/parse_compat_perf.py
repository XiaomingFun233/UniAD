#!/usr/bin/env python3
import re
import statistics
import sys
from pathlib import Path

PATTERN = re.compile(
    r"\[CompatRunnerPerf\]\s+epoch=(?P<epoch>\d+)/(?:\d+)\s+iter=(?P<iter>\d+)\s+global_iter=(?P<global_iter>\d+)\s+"
    r"loss=(?P<loss>[-+eE0-9\.]+)\s+data_time=(?P<data_time>[-+eE0-9\.]+)s\s+iter_time=(?P<iter_time>[-+eE0-9\.]+)s\s+"
    r"throughput=(?P<throughput>[-+eE0-9\.]+)\s+samples/s\s+avg_throughput=(?P<avg_throughput>[-+eE0-9\.]+)\s+samples/s\s+"
    r"global_batch=(?P<global_batch>\d+)"
)


def _mean(vals):
    return statistics.fmean(vals) if vals else 0.0


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: parse_compat_perf.py <log_path> [warmup_global_iter]")
        return 2
    log_path = Path(sys.argv[1])
    warmup_global_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}")
        return 2

    points = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = PATTERN.search(line)
        if not m:
            continue
        d = m.groupdict()
        points.append(
            {
                "epoch": int(d["epoch"]),
                "iter": int(d["iter"]),
                "global_iter": int(d["global_iter"]),
                "loss": float(d["loss"]),
                "data_time": float(d["data_time"]),
                "iter_time": float(d["iter_time"]),
                "throughput": float(d["throughput"]),
                "avg_throughput": float(d["avg_throughput"]),
                "global_batch": int(d["global_batch"]),
            }
        )

    if not points:
        print(f"ERROR: no [CompatRunnerPerf] entries found in {log_path}")
        return 1

    all_tp = [p["throughput"] for p in points]
    all_it = [p["iter_time"] for p in points]
    all_dt = [p["data_time"] for p in points]

    stable = [p for p in points if p["global_iter"] > warmup_global_iter]
    if not stable:
        stable = points

    st_tp = [p["throughput"] for p in stable]
    st_it = [p["iter_time"] for p in stable]
    st_dt = [p["data_time"] for p in stable]

    final = points[-1]
    print(f"log: {log_path}")
    print(f"global_batch: {final['global_batch']}")
    print(f"points: {len(points)}")
    print(f"throughput_avg_all: {_mean(all_tp):.2f} samples/s")
    print(f"throughput_median_all: {statistics.median(all_tp):.2f} samples/s")
    print(f"iter_time_avg_all: {_mean(all_it):.4f}s")
    print(f"data_time_avg_all: {_mean(all_dt):.4f}s")
    print(f"throughput_avg_stable(global_iter>{warmup_global_iter}): {_mean(st_tp):.2f} samples/s")
    print(f"iter_time_avg_stable: {_mean(st_it):.4f}s")
    print(f"data_time_avg_stable: {_mean(st_dt):.4f}s")
    print(
        "last_point: "
        f"epoch={final['epoch']} iter={final['iter']} global_iter={final['global_iter']} "
        f"loss={final['loss']:.6f} throughput={final['throughput']:.2f} samples/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
