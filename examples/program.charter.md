# autoresearch — charter edition

A drop-in replacement for the baseline `program.md` that adds **comprehensive goal
setting** and **persistent memory**. Same operational loop; sharper direction.

**How to use:** copy this file to the repo root as `program.md` (back up the
original first), copy `examples/LESSONS.md` to the repo root, then point your agent
at it exactly as before. Edit the charter to match your goals.

---

## Research charter

Read this charter at the start of every experiment. It defines what "good" means,
what you may not touch, and how to get unstuck.

**North star:** minimize `val_bpb` on the fixed 5-minute budget.

**Hard constraints (never violate):**
- `peak_vram_mb` must stay under 60000 — treat OOM as a `discard`, not a bug to chase.
- `prepare.py` is read-only. `evaluate_bpb` is the ground-truth metric; never touch it.
- Change only one substantive thing per experiment, so results stay attributable.

**Milestones (advance in order; record when each is hit):**
1. **M1** — reproduce and record the baseline (unmodified `train.py`).
2. **M2** — beat baseline by ≥ 1% `val_bpb`.
3. **M3** — hold M2's `val_bpb` while cutting `peak_vram_mb` by 10%.
4. **M4** — beat M2 by another 1%, by any means.

**Idea-selection priorities (when choosing the next experiment):**
1. Prefer changes with a clear mechanistic hypothesis over blind sweeps.
2. Prefer simplifications that hold the metric (delete-and-win).
3. Revisit the two best `discard`ed ideas in `results.tsv` and try combining them.
4. When a category stalls, move to the next (see pivot rule).

**Pivot / stop:**
- If 15 experiments pass with no improvement, switch category in this order:
  architecture → optimizer → schedule → data-packing knobs.
- Never stop on your own. Run until interrupted.

## Setup

1. **Agree on a run tag** (e.g. `mar5`). The branch `autoresearch/<tag>` must not exist yet.
2. **Create the branch:** `git checkout -b autoresearch/<tag>` from master.
3. **Read the in-scope files:** `README.md`, `prepare.py` (do not modify), `train.py` (you modify this).
4. **Verify data exists:** check `~/.cache/autoresearch/` has data shards + a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize `results.tsv`** with just the header row: `commit	val_bpb	memory_gb	status	description`.
6. **Read `LESSONS.md`** to load prior knowledge. If it doesn't exist, create it from `examples/LESSONS.md`.
7. **Confirm** setup looks good, then begin.

## Memory protocol

- **Before** each experiment: re-read `LESSONS.md`. Do not retry anything under "Dead ends".
- **After** each experiment: append one line to the right section — a failed idea goes to "Dead ends", a partial win to "Live leads", a kept change to "Confirmed wins". Keep it factual and reference the experiment.

## The experiment loop

The experiment runs on your dedicated branch. Each run has a fixed 5-minute budget.

LOOP FOREVER:

1. Look at the git state (current branch/commit) and re-read the charter + `LESSONS.md`.
2. Pick the next experiment using the idea-selection priorities. Tune `train.py`.
3. `git commit` the change.
4. Run it: `uv run train.py > run.log 2>&1` (redirect everything; do not flood context).
5. Read the result: `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
6. If the grep is empty, the run crashed — `tail -n 50 run.log`, and either fix a trivial bug and re-run, or log a `crash` and move on.
7. Record the result in `results.tsv` (leave it untracked by git).
8. Update `LESSONS.md`.
9. **Keep or discard:** if `val_bpb` improved (lower), keep the commit and advance. If equal or worse, `git reset` back to where you started.

**Timeout:** each run should take ~5 minutes (+ startup/eval). If a run exceeds 10 minutes, kill it and treat it as a failure.

## Output format & logging

The script prints a summary ending with `val_bpb:`, `peak_vram_mb:`, etc. Log each experiment to `results.tsv` (tab-separated) with columns:

```
commit	val_bpb	memory_gb	status	description
```

- `val_bpb`: achieved value, or `0.000000` for a crash.
- `memory_gb`: `peak_vram_mb / 1024`, one decimal; `0.0` for a crash.
- `status`: `keep`, `discard`, or `crash`.
- `description`: one short line of what the experiment tried.

Check progress any time with `uv run examples/dashboard.py`.
