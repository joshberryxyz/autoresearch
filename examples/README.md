# Starter kit

Runnable companions to the [continuous-research-org blueprints](../README.md#scaling-up-a-continuous-research-org)
in the main README. These are optional add-ons; the core project is still just
the four files in the repo root. Copy what you need and edit freely.

| File | Blueprint | What it is | How to use |
|------|-----------|------------|------------|
| [`program.charter.md`](program.charter.md) | #1 goal setting, #2 memory | A drop-in `program.md` with a research charter (north star, constraints, milestones, priorities, pivot rules) and a read/append memory protocol. | Back up `program.md`, then copy this to the repo root as `program.md`. |
| [`LESSONS.md`](LESSONS.md) | #2 memory | A lab-notebook template the agent reads before and appends after each experiment, so knowledge compounds across sessions. | Copy to the repo root; delete the example entries. |
| [`program.swarm.md`](program.swarm.md) | #3 swarm, #4 evolution | Roles + a principal-investigator protocol for running several specialist agents in parallel on separate branches. | Launch one agent per role; one agent (or you) plays PI. |
| [`dashboard.py`](dashboard.py) | #7 operations | A standalone status dashboard over `results.tsv`: keep-rate, champion, and a text frontier sparkline. Can also emit a Markdown "morning report" and a PNG plot. | `uv run examples/dashboard.py` |

## dashboard.py

```bash
uv run examples/dashboard.py                     # text dashboard (reads ./results.tsv)
uv run examples/dashboard.py --tsv path/to.tsv   # a specific results file (e.g. a swarm branch)
uv run examples/dashboard.py --report report.md  # also write a Markdown morning report
uv run examples/dashboard.py --plot frontier.png # also save a frontier plot
```

Example output:

```
autoresearch dashboard
==========================================
experiments : 6   keep 3 · discard 2 · crash 1
keep-rate   : 60%
baseline    : 0.997900 val_bpb
best        : 0.991000 val_bpb
improvement : 0.006900 (0.69%)
champion    : 0.991000  value-emb gate channels 32->64  [e5f6a7b]

frontier    : █▃▃▁▁
```

It uses only `pandas` and `matplotlib`, which are already project dependencies, so
`uv run` needs no extra setup.
