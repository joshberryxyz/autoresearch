# autoresearch — swarm edition

A template for running **several specialist agents in parallel**, each on its own
branch with a focused mandate, coordinated by a lightweight "principal
investigator" (PI) protocol. Requires multiple GPUs (or multiple time slices on
one). Builds on the charter edition — read that first.

**How to use:** launch one agent per role below, each in its own worktree/checkout,
and give each the same run tag. One agent (or the human) plays PI.

---

## Roles

Pick **one** role per agent. Each role is a lens; all share the same north star
(lowest `val_bpb`) but explore a different part of the space, so they rarely
collide.

| Role | Branch | Mandate — only touch these areas of `train.py` |
|------|--------|-----------------------------------------------|
| `arch` | `autoresearch/arch-<tag>` | Architecture: norms, activations, attention variants, value embeddings, residual mixing, window pattern. |
| `opt` | `autoresearch/opt-<tag>` | Optimization: Muon/AdamW learning rates, betas, weight decay, momentum, schedules. |
| `eff` | `autoresearch/eff-<tag>` | Efficiency: batch size, grad-accum, depth/width trade-offs, anything that buys more steps in the budget. |
| `wild` | `autoresearch/wild-<tag>` | Wildcard: radical or unconventional ideas that the other roles would consider too risky. |

Each agent runs the **charter-edition loop** (`examples/program.charter.md`),
restricted to its mandate, and logs to its own `results.tsv` on its own branch.

## Principal investigator (PI) protocol

Run this periodically (e.g. every 60–90 minutes, or every N experiments per agent):

1. **Collect.** For each role branch, read the best (lowest `val_bpb`) `keep`
   entry from that branch's `results.tsv`.
2. **Crown the champion.** The global champion is simply the branch holding the
   lowest `val_bpb` — one comparable number makes this trivial.
3. **Merge wins.** Cherry-pick or re-apply the champion's winning change onto a
   shared `autoresearch/champion-<tag>` branch. When two roles' wins are
   independent (e.g. an optimizer tweak and an architecture tweak), try combining
   them and verify the combination actually improves on each alone.
4. **Broadcast.** Append confirmed cross-role wins to a shared `LESSONS.md` so
   every agent benefits.
5. **Reassign.** If a role has stalled (see the charter's pivot rule), narrow or
   widen its mandate, or point it at combining another role's near-misses.

## Notes

- Keep branches isolated. Agents should never edit each other's branch; the PI is
  the only writer to the shared champion branch.
- `uv run examples/dashboard.py --tsv <branch results.tsv>` gives a per-role view;
  run it across branches to compare frontiers.
- This is the simplest useful swarm. Natural extensions: a rotating PI, a shared
  idea queue, or an evolutionary layer that forks the champion into new mutations.
