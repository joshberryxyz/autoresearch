#!/usr/bin/env python3
"""
examples/dashboard.py: a standalone status dashboard for an autoresearch run.

Reads results.tsv (the experiment log described in program.md) and prints a
summary of research progress: how many experiments ran, the keep-rate, the
current champion, and the descending val_bpb "frontier". Optionally writes a
Markdown "morning report" and/or a PNG plot of the frontier.

Usage:
    uv run examples/dashboard.py                        # text dashboard (reads ./results.tsv)
    uv run examples/dashboard.py --tsv path/to/results.tsv
    uv run examples/dashboard.py --report report.md     # also write a Markdown report
    uv run examples/dashboard.py --plot frontier.png    # also save a frontier plot

results.tsv is tab-separated with columns:
    commit  val_bpb  memory_gb  status  description
"""
import argparse
import os
import sys

import pandas as pd

SPARK = "▁▂▃▄▅▆▇█"  # short = better (low bpb), tall = worse; frontier should trend down


def load(tsv_path):
    if not os.path.exists(tsv_path):
        sys.exit(
            f"No results file at {tsv_path!r}. Run some experiments first "
            f"(see program.md), or pass --tsv <path>."
        )
    df = pd.read_csv(tsv_path, sep="\t")
    for col in ("val_bpb", "memory_gb"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = df["status"].astype(str).str.strip().str.upper()
    return df


def summarize(df):
    counts = df["status"].value_counts()
    n_keep = int(counts.get("KEEP", 0))
    n_discard = int(counts.get("DISCARD", 0))
    n_crash = int(counts.get("CRASH", 0))
    decided = n_keep + n_discard

    valid = df[df["status"] != "CRASH"].reset_index(drop=True)
    baseline = float(valid["val_bpb"].iloc[0]) if len(valid) else float("nan")
    frontier = valid["val_bpb"].cummin() if len(valid) else valid["val_bpb"]
    best = float(frontier.min()) if len(valid) else float("nan")

    kept = df[df["status"] == "KEEP"]
    champion = kept.loc[kept["val_bpb"].idxmin()] if len(kept) else None

    return {
        "n": len(df),
        "n_keep": n_keep,
        "n_discard": n_discard,
        "n_crash": n_crash,
        "keep_rate": (n_keep / decided) if decided else 0.0,
        "baseline": baseline,
        "best": best,
        "improvement": (baseline - best) if len(valid) else float("nan"),
        "champion": champion,
        "frontier": frontier,
        "valid": valid,
    }


def sparkline(values):
    vals = [v for v in values if v == v]  # drop NaN
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return SPARK[-1] * len(vals)
    return "".join(SPARK[min(int((v - lo) / (hi - lo) * (len(SPARK) - 1)), len(SPARK) - 1)] for v in vals)


def render_text(s):
    lines = []
    lines.append("autoresearch dashboard")
    lines.append("=" * 42)
    lines.append(f"experiments : {s['n']}   keep {s['n_keep']} · discard {s['n_discard']} · crash {s['n_crash']}")
    lines.append(f"keep-rate   : {s['keep_rate']:.0%}")
    if s["baseline"] == s["baseline"]:
        lines.append(f"baseline    : {s['baseline']:.6f} val_bpb")
        lines.append(f"best        : {s['best']:.6f} val_bpb")
        pct = (s["improvement"] / s["baseline"] * 100) if s["baseline"] else 0.0
        lines.append(f"improvement : {s['improvement']:.6f} ({pct:.2f}%)")
    if s["champion"] is not None:
        c = s["champion"]
        lines.append(f"champion    : {c['val_bpb']:.6f}  {str(c['description']).strip()}  [{str(c['commit']).strip()}]")
    spark = sparkline(list(s["frontier"]))
    if spark:
        lines.append("")
        lines.append(f"frontier    : {spark}")
        lines.append("              (each cell = an experiment; lower is better, so it should trend down)")
    return "\n".join(lines)


def render_markdown(s):
    lines = ["# autoresearch: morning report", ""]
    lines.append(f"- **Experiments:** {s['n']} (keep {s['n_keep']} · discard {s['n_discard']} · crash {s['n_crash']})")
    lines.append(f"- **Keep-rate:** {s['keep_rate']:.0%}")
    if s["baseline"] == s["baseline"]:
        pct = (s["improvement"] / s["baseline"] * 100) if s["baseline"] else 0.0
        lines.append(f"- **Baseline → best:** {s['baseline']:.6f} → {s['best']:.6f} val_bpb ({pct:.2f}% better)")
    if s["champion"] is not None:
        c = s["champion"]
        lines.append(f"- **Champion:** {str(c['description']).strip()} (`{str(c['commit']).strip()}`, {c['val_bpb']:.6f})")
    spark = sparkline(list(s["frontier"]))
    if spark:
        lines.append("")
        lines.append(f"Frontier: `{spark}`")
    return "\n".join(lines) + "\n"


def save_plot(df, s, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required for --plot (it's already a project dependency; run via `uv run`).")

    valid = s["valid"]
    fig, ax = plt.subplots(figsize=(12, 6))
    disc = valid[valid["status"] == "DISCARD"]
    kept = valid[valid["status"] == "KEEP"]
    ax.scatter(disc.index, disc["val_bpb"], c="#c7ccd6", s=14, alpha=0.6, label="discarded", zorder=2)
    ax.scatter(kept.index, kept["val_bpb"], c="#0c8b82", s=36, label="kept", zorder=3)
    ax.plot(valid.index, s["frontier"], c="#a96b14", lw=2, label="frontier (running min)", zorder=1)
    ax.set_xlabel("experiment")
    ax.set_ylabel("val_bpb (lower is better)")
    ax.set_title("autoresearch val_bpb frontier")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved plot → {path}")


def main():
    p = argparse.ArgumentParser(description="Status dashboard for an autoresearch run.")
    p.add_argument("--tsv", default="results.tsv", help="path to results.tsv (default: ./results.tsv)")
    p.add_argument("--report", metavar="FILE", help="also write a Markdown morning report to FILE")
    p.add_argument("--plot", metavar="FILE", help="also save a PNG frontier plot to FILE")
    args = p.parse_args()

    df = load(args.tsv)
    s = summarize(df)
    print(render_text(s))

    if args.report:
        with open(args.report, "w") as f:
            f.write(render_markdown(s))
        print(f"\nwrote report → {args.report}")
    if args.plot:
        save_plot(df, s, args.plot)


if __name__ == "__main__":
    main()
