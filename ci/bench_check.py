#!/usr/bin/env python3
"""Compare an `ocak bench` Markdown table against a stored baseline.

usage: bench_check.py BASELINE.md CURRENT.md [CURRENT2.md ...]

With several CURRENT tables (repeat runs), each model's best throughput is used, so a transient
slowdown (another process using the GPU during one run) does not fail the check.
- A model whose row is ERR (load/generate failure) fails the check.
- Prefill or decode throughput more than WARN_PCT below the baseline emits a GitHub
  warning annotation; more than FAIL_PCT below fails the check.
- Models missing from either side are reported but do not fail (the model set on the
  runner can change). Refresh the baseline by copying a known-good run's table over it.
"""
import sys

WARN_PCT = 10.0
FAIL_PCT = 25.0


def parse(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 7 or cells[0] in ("Model", "") or cells[0].startswith("---"):
                continue
            model, load, prefill, decode = cells[0], cells[3], cells[4], cells[5]
            if load == "ERR":
                rows[model] = None
                continue
            try:
                rows[model] = (float(prefill), float(decode))
            except ValueError:
                rows[model] = None
    return rows


def main():
    base = parse(sys.argv[1])
    cur = {}
    for path in sys.argv[2:]:
        for model, vals in parse(path).items():
            prev = cur.get(model)
            if vals is None:
                cur.setdefault(model, None)
            elif prev is None:
                cur[model] = vals
            else:
                cur[model] = (max(prev[0], vals[0]), max(prev[1], vals[1]))
    failed = False
    out = ["| Model | Prefill base → now | Decode base → now |", "|---|---:|---:|"]
    for model, vals in cur.items():
        if vals is None:
            print(f"::error::{model}: bench row is ERR")
            out.append(f"| {model} | ERR | ERR |")
            failed = True
            continue
        b = base.get(model)
        if not b:
            out.append(f"| {model} | (new) {vals[0]:.1f} | (new) {vals[1]:.1f} |")
            continue
        cells = []
        for name, bv, cv in (("prefill", b[0], vals[0]), ("decode", b[1], vals[1])):
            delta = 100.0 * (cv - bv) / bv if bv > 0 else 0.0
            cells.append(f"{bv:.1f} → {cv:.1f} ({delta:+.0f}%)")
            if delta < -FAIL_PCT:
                print(f"::error::{model}: {name} {cv:.1f} tok/s is {-delta:.0f}% below baseline {bv:.1f}")
                failed = True
            elif delta < -WARN_PCT:
                print(f"::warning::{model}: {name} {cv:.1f} tok/s is {-delta:.0f}% below baseline {bv:.1f}")
        out.append(f"| {model} | {cells[0]} | {cells[1]} |")
    for model in base:
        if model not in cur:
            # A model in the baseline must be benchmarked: an empty or truncated bench output
            # (crash, missing model file) is a failure, not a pass.
            print(f"::error::{model}: missing from the benchmark output")
            out.append(f"| {model} | (missing) | (missing) |")
            failed = True
    print("\n".join(out))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
