# Results

Every number here was measured on 19 protein chains. Each modification ran on the same random seeds as the baseline, so the comparisons are paired, and the confidence intervals come from a bootstrap over the 19 chains.

The raw data is in [`results/`](../results/). Any table can be regenerated with:

```bash
python experiments/analyze_experiments.py --csv results/experiments_19chains.csv
```

## Setup

The 19 chains range from 46 to 436 residues with a median of 214. I pulled them from PDB files bundled inside ProteinMPNN, ProDy, Biopython and pdbfixer, then filtered to chains with complete backbone coordinates, no unknown residues and no low-complexity sequences, and dropped duplicate sequences. Each chain was extracted to its own PDB file so that every design task is a single chain.

Chains that originally came from complexes are designed without their binding partners, so their recovery is lower than a true monomer would be. This does not affect the comparisons, because every modification sees the same structures.

The model is `v_48_020.pt`, the standard vanilla ProteinMPNN weights. The metric is native sequence recovery, the fraction of positions matching the native sequence. Negative log likelihood is reported alongside it where it is informative.

The unit of analysis is the structure rather than the trial. Trials on the same backbone are not independent evidence that a method works, so recovery is averaged within a structure first, and then the per-structure differences against the baseline are summarized across structures. The confidence interval is a bootstrap over those 19 differences.

### A seeding bug that affected the first runs

ProteinMPNN draws amino acids using `torch.multinomial`, which reads PyTorch's global random number generator. My first version of `Runner.sample` seeded only the NumPy generator, so the seed fixed the decoding order and left the amino acid draws unseeded. Two calls with the same seed returned different sequences.

I noticed because 18 of 19 baseline values disagreed between two runs that should have been identical. `Runner.sample` now derives a PyTorch seed from the generator it is passed, and a regression test holds that property.

Both arms of every comparison received independent sampling noise, so the estimates were unbiased, but pairing only removed the decoding-order variance and not the amino acid variance. The intervals were wider and the t statistics more conservative than they should have been. Nothing was inflated. Where it matters I say so below, and the headline consensus number comes from the corrected run.

## What worked

### The decoding rule, improved by 1.23% for one forward pass

Four estimators of the same model, on the same 19 chains, with 4 paired trials each:

| Estimator | Model calls | Recovery | Gain | 95% CI | t | Better on |
|---|---|---|---|---|---|---|
| Ancestral sampling, the default | 1 | 45.61% | | | | |
| Greedy, approximating the joint mode | 1 | 45.63% | +0.02% | [-0.49%, +0.50%] | +0.07 | 9 of 19 |
| Unconditional marginal | 1 | 46.84% | +1.23% | [+0.09%, +2.38%] | +2.03 | 14 of 19 |
| Order-averaged marginal | 32 | 47.12% | +1.51% | [+0.48%, +2.37%] | +3.04 | 17 of 19 |

This is the central result, and Section 2 of [`theory.md`](theory.md) predicts all of it. Recovery is Hamming loss, the estimator that minimizes Hamming loss is the per-position marginal mode, and neither sampling nor the joint mode is that estimator.

Three predictions and three confirmations. Both marginal estimators beat sampling. The greedy joint mode does not, because it is optimal for a stricter loss that nobody reports. Compared head to head, the order-averaged version beats the unconditional one by only 0.28%, with a confidence interval of [-0.84%, +1.44%] and a t of 0.46, which is not a distinguishable difference.

The practical consequence is that ProteinMPNN already contains a marginal decoder. `unconditional_probs` has been in `protein_mpnn_utils.py` since release. It costs one forward pass, it is deterministic, and taking its argmax improves recovery by 1.23% over the sampling path that recovery is normally computed from:

```python
log_probs = model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)
sequence = log_probs[0].argmax(dim=-1)
```

The library already had the better decoder for this metric. What was missing was the connection between that function and the number being reported.

One caveat matters here. This is measured on recovery alone. Marginal decoding maximizes per-position agreement and makes no promise that the sequence is coherent as a whole, because it never asks whether the chosen amino acids fit together. If the objective were foldability or stability, the loss would be different and so would the optimal estimator. That is the point of Section 2 of the theory document, and it cuts both ways.

### Order-marginalized decoding, improved by 1.51%

From the corrected run in `consensus_paired.csv`, with 4 replicates and proper seeding:

| Estimator | Recovery | Gain | 95% CI | t | Better on |
|---|---|---|---|---|---|
| Baseline at T=0.1 | 45.32% | | | | |
| Order-averaged, N=32 | 46.94% | +1.62% | [+0.77%, +2.37%] | +3.85 | 17 of 19 |
| Order-averaged, N=8 | 46.69% | +1.37% | [+0.64%, +1.97%] | +3.89 | 16 of 19 |
| Greedy at T=0.001 | 45.55% | +0.23% | [-0.17%, +0.63%] | +1.11 | 12 of 19 |

Earlier runs made before the seeding fix put the N=8 figure anywhere between +0.47% and +1.21%. That spread was the unseeded sampling noise. The table above is the one to use.

Each sample's `probs` output holds the conditional distribution the model used at every position, under that sample's own ordering and prefix. Averaging those across orderings estimates the marginal with both the prefix and the ordering integrated out. Taking the argmax of that average gives the marginal mode.

Cell 31 of the original notebook attempted something close to this and measured no effect. The reason is in [`findings.md`](findings.md). After building the consensus it passed the result into another `sample` call with an all-ones chain mask, and that call discarded it.

Sample count matters with diminishing returns. The pre-fix sweep across 19 structures gives +0.47% at N=8, +0.61% at N=16, +0.83% at N=32 and +1.01% at N=64. Those absolute values are depressed by the seeding bug but the trend is monotone and roughly logarithmic. In the corrected run, moving from N=8 to N=32 raised the gain from +1.37% to +1.62%, so most of the benefit arrives by N=8 and N=32 is a reasonable place to stop.

### Fixing the neighbour count, improved by 2.04%

| Arm | Recovery | Gain | 95% CI | t | Better on |
|---|---|---|---|---|---|
| k=48, read from the checkpoint | 45.65% | | | | |
| k=64, the original configuration | 43.62% | -2.04% | [-3.01%, -1.17%] | -4.20 | 2 of 19 |

The original notebook built the model with `k_neighbors=64` and loaded `v_48_020.pt`. The 48 in that filename is the neighbour count the weights were trained with.

`k_neighbors` does not change any parameter shapes, so `load_state_dict` succeeds and nothing warns. The model runs on a 64-neighbour graph it has never seen. Every recovery number collected in the original notebook was depressed by about two points because of this.

It also costs 0.11 nats of negative log likelihood, 0.9288 against 1.0395, which is the largest likelihood degradation of any modification I tested. `ipr.backends.load_model` now reads `checkpoint['num_edges']` instead of taking a hardcoded value.

### Lowering the temperature, improved by 0.32%

| Temperature | Gain | 95% CI | t |
|---|---|---|---|
| 0.03 | +0.32% | [+0.04%, +0.59%] | +2.22 |
| 0.01 | +0.32% | [+0.07%, +0.57%] | +2.40 |
| 0.3 | -1.44% | [-2.20%, -0.75%] | -3.83 |
| 0.001 | -0.66% | [-1.27%, -0.05%] | -2.03 |

Going lower than the usual 0.1 helps a little. Going all the way to greedy hurts, so the best setting sits around 0.03 rather than at zero.

## What did not work

### Second-pass refinement

| Arm | Gain | 95% CI | t |
|---|---|---|---|
| Entropy-guided refinement | -0.33% | [-0.88%, +0.18%] | -1.21 |
| Random selection, matched budget | -0.22% | [-0.85%, +0.36%] | -0.68 |

This was the hypothesis the project started from, and it does not hold up.

The version in the original notebook never conditioned on its draft, because of the chain-mask bug described in [`findings.md`](findings.md). That bug is fixed here. `ipr.refine.uncertainty_guided` redesigns the least confident residues while genuinely holding the rest as context, which is what the project set out to build.

Run properly on 19 structures it comes out slightly negative, and it never beats the matched-budget random control. Choosing which residues to revisit by entropy does no better than choosing them at random, which means the model's uncertainty does not carry information about where it is wrong.

The result therefore stands as a negative one rather than an artifact. The idea was reasonable, the implementation is now correct, and it still does not improve recovery.

### Best-of-N selection by likelihood

| Arm | Gain | 95% CI | t |
|---|---|---|---|
| Best of 4 by NLL | +0.20% | [-0.50%, +0.99%] | +0.51 |

Drawing four samples and keeping the one the model scores best improves negative log likelihood from 0.9288 to 0.9055 and does essentially nothing for recovery.

This is worth separating from the tournament cell in the original notebook, which kept the best sample by recovery against the native sequence. That is selecting on the answer, and a number produced that way cannot be reported as a method result.

### Decoding order

| Arm | Gain | 95% CI | t |
|---|---|---|---|
| Buried residues first | +0.56% | [-0.17%, +1.22%] | +1.55 |
| Exposed residues first | -0.63% | [-1.66%, +0.44%] | -1.14 |
| N-terminus to C-terminus | -0.76% | [-1.55%, -0.03%] | -1.89 |

Neither structural ordering clears significance, but the signs split the way the project notes predicted. Decoding buried residues first helps a little, decoding exposed residues first hurts a little, and a fixed N-to-C order is worse than random. The effect is small enough that settling it would need more structures.

## Likelihood and recovery come apart

This deserves stating on its own, because it affects how everything above should be read.

Entropy-guided refinement posted the second-best negative log likelihood of any modification, 0.9036 against the baseline's 0.9288, while scoring worse on recovery. Best-of-4 improved likelihood and not recovery. Both are producing sequences the model prefers and the native sequence agrees with less.

If you plan to optimize likelihood and expect recovery to follow, check that assumption on your own data first.

## Full table

19 structures, 8 paired trials each. The baseline is k=48 at T=0.1.

| Arm | Recovery | Gain | 95% CI | t | Better on | NLL |
|---|---|---|---|---|---|---|
| Baseline | 45.65% | | | | | 0.9288 |
| Buried residues first | 46.21% | +0.56% | [-0.17%, +1.22%] | +1.55 | 63% | 0.9264 |
| T=0.01 | 45.97% | +0.32% | [+0.07%, +0.57%] | +2.40 | 68% | 0.9163 |
| T=0.03 | 45.97% | +0.32% | [+0.04%, +0.59%] | +2.22 | 68% | 0.9164 |
| Best of 4 | 45.86% | +0.20% | [-0.50%, +0.99%] | +0.51 | 47% | 0.9055 |
| Random refinement | 45.44% | -0.22% | [-0.85%, +0.36%] | -0.68 | 47% | 0.9218 |
| Entropy refinement | 45.32% | -0.33% | [-0.88%, +0.18%] | -1.21 | 42% | 0.9036 |
| Exposed residues first | 45.02% | -0.63% | [-1.66%, +0.44%] | -1.14 | 37% | 0.9361 |
| N to C order | 44.90% | -0.76% | [-1.55%, -0.03%] | -1.89 | 37% | 0.9272 |
| T=0.3 | 44.21% | -1.44% | [-2.20%, -0.75%] | -3.83 | 21% | 1.0063 |
| k=64 | 43.62% | -2.04% | [-3.01%, -1.17%] | -4.20 | 11% | 1.0395 |

## Limitations

Nineteen structures assembled from whatever PDB files happened to be bundled in a handful of Python packages is a pilot rather than a benchmark. It is not a curated or representative set. Four of the chains are modified structures, one with added hydrogens, one a normal-mode perturbation and one an alignment-fixed model. The proper version of this experiment uses ProteinMPNN's own held-out test split.

Chains taken from complexes are designed without their binding partners, so their interface residues have no context and their recovery is dragged down. This applies equally to every modification, so the paired comparisons remain valid, but the absolute numbers are lower than they would be on true monomers.

Everything ran on CPU with a single checkpoint. I did not test whether these effects hold for the soluble or CA-only weights.

The sample-count scaling table predates the seeding fix, so its absolute values are depressed and only the trend should be read from it. The headline consensus number comes from the corrected run.

## Reproducing

```bash
./scripts/setup_proteinmpnn.sh

python experiments/build_benchmark_set.py \
    --search ProteinMPNN/inputs <other directories containing PDB files> \
    --min-length 45 --max-length 450 --out benchmark_set.json

python experiments/run_experiments.py --set benchmark_set.json \
    --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \
    --trials 8 --out results/experiments_19chains.csv

python experiments/run_consensus.py --set benchmark_set.json \
    --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \
    --n-samples-list 8 32 --reps 4 --temperatures 0.5 \
    --out results/consensus_paired.csv

python experiments/analyze_experiments.py --csv results/experiments_19chains.csv
```

The full sweep takes about 47 minutes on CPU, the consensus run about 35, and the estimator comparison about 24.
