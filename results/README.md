# Results

Raw output from the experiments. Analysis and interpretation are in
[`docs/results.md`](../docs/results.md).

| File | What it holds |
|---|---|
| `experiments_19chains.csv` | Main sweep. 11 arms, 8 paired trials, 19 chains. 1558 rows. |
| `consensus_19chains.csv` | Marginal consensus decoding at T=0.3, 0.5, 0.8 with N=8. |
| `consensus_nsweep.csv` | Sample-count scaling at T=0.5, N in {8,16,32,64}. Predates the seeding fix, read the trend only. |
| `consensus_paired.csv` | Corrected consensus run, N in {8,32}, 4 reps. |
| `estimators_19chains.csv` | Four decoding estimators compared: ancestral, joint mode, unconditional marginal, order-averaged marginal. **The central result.** |

Each row is one observation:

```
structure,length,arm,trial,recovery,nll,n_changed
```

## Regenerating the tables

```bash
python experiments/analyze_experiments.py --csv results/experiments_19chains.csv
python experiments/analyze_experiments.py --csv results/consensus_19chains.csv
python experiments/analyze_experiments.py --csv results/consensus_nsweep.csv
```

The analysis averages trials within a structure, takes per-structure paired
differences against the baseline arm, and bootstraps a confidence interval over
structures. The structure is the unit of analysis because trials on the same
backbone are not independent evidence that a method works.

## Run configuration

All files come from `v_48_020.pt` on CPU, against the 19 chains listed in
`benchmark_set.json`.

`consensus_paired.csv` was produced after a seeding fix in `Runner.sample`.
The earlier files were generated while ProteinMPNN's token sampling ran off
torch's unseeded global RNG, so their arms are paired on decoding order but not
on token draws. Their estimates are unbiased, their intervals are too wide.
See the reproducibility note in [`docs/results.md`](../docs/results.md).

| | |
|---|---|
| Weights | `ProteinMPNN/vanilla_model_weights/v_48_020.pt` |
| Neighbour count | 48, read from `checkpoint['num_edges']` |
| Backbone noise | 0.0 |
| Baseline | single pass, T=0.1 |
| Seeds | 0..7 for the main sweep, 1000+ for consensus |
| Runtime | 47 min main sweep, 11 min consensus, 45 min N sweep |

## Older numbers

The results recorded in the original Colab notebook are kept separately in
[`notes/notebook_results.csv`](../notes/notebook_results.csv), since they came
from a single protein under a misconfigured model. Summarise them with:

```bash
python experiments/analyze_notebook_results.py
```

## Adding to this directory

`.gitignore` excludes `results/*.csv` by default, so committing a result is a
deliberate act:

```bash
git add -f results/<name>.csv
```

When you add one, record the weights file, the structure set, trial count,
seed and the exact command. A recovery number without its sample size is not
interpretable.
