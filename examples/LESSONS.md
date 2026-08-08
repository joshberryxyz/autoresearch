# Lessons — the lab notebook

> **Protocol:** read this file *before* choosing each experiment, and append to it
> *after* each one. `results.tsv` records *what* happened; this file records *why*,
> so knowledge compounds across sessions instead of being rediscovered every night.
>
> Keep entries short and factual. Reference experiments by their `results.tsv`
> commit or a `#index`. Delete the example entries below once you have real ones.

## Dead ends — do not retry

*Ideas that were tested and clearly failed. Avoid repeating them.*

- _(example)_ GeLU activation: consistently +0.004 val_bpb vs squared-ReLU (#12, #29)
- _(example)_ matrix_lr > 0.06: diverges within ~200 steps, NaN loss (#7)

## Live leads — promising, revisit

*Ideas that looked good but were noisy, or partial wins worth pushing further.*

- _(example)_ value-embedding gate channels 32 → 64: −0.001 but noisy; retry with more steps
- _(example)_ shorter windows freed throughput but hurt long-range; try `SSSL` → `SSSSL`

## Confirmed wins — currently in the champion

*Changes that stuck and are part of the current best config.*

- _(example)_ matrix_lr 0.04 → 0.045 (−0.002, #18)

## Open questions

*Things you don't understand yet and want to investigate.*

- _(example)_ why does depth 9 underperform both 8 and 10 in the fixed budget?
