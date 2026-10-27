# ProteinMPNN Decoding Analysis

ProteinMPNN is a protein-sequence design model that generates an amino acid sequence likely to fold to a given protein backbone structure. It relies on autoregressive decoding, placing amino acids one at a time conditioning on only those amino acids previously placed. Amino acids placed early in the sequence condition on little information, and there is no second-pass of the sequence to validate them. Given the long-range dependencies of amino acids in a protein, I found these aspects of autoregressive decoding unintuitive. I tested various modifications to autoregressive decoding to address these shortcomings, resulting in a 1.5% improvement in native sequence recovery (NSR), the primary metric for ProteinMPNN performance. Changing a separate configuration of autoregressive decoding improved NSR another 2%.

## Results

I tested every modification on 19 protein chains ranging from 46 to 436 residues. Each modification ran on the same random seeds as the baseline so the comparison is paired, and the confidence intervals come from a bootstrap over the 19 chains.

| Modification | Cost | NSR gain | 95% CI | Better on |
|---|---|---|---|---|
| Correcting the neighbour count | free | +2.04% | [+1.17%, +3.01%] | 17 of 19 chains |
| Order-marginalized decoding | 32x | +1.51% | [+0.48%, +2.37%] | 17 of 19 chains |
| Marginal decoding, one forward pass | free | +1.23% | [+0.09%, +2.38%] | 14 of 19 chains |
| Lowering the sampling temperature | free | +0.32% | [+0.04%, +0.59%] | 13 of 19 chains |
| Greedy decoding | free | +0.02% | [-0.49%, +0.50%] | 9 of 19 chains |
| Second-pass refinement | 3x | -0.33% | [-0.88%, +0.18%] | 8 of 19 chains |

The baseline is standard ProteinMPNN sampling at temperature 0.1, which recovers 45.6% of native residues on this set.

## Why the decoding procedure is mismatched to the metric

NSR counts how many individual positions match the native sequence. Every residue is graded on its own, and getting one right does not depend on getting its neighbours right.

ProteinMPNN returns a sequence sampled from its joint distribution. That sequence is plausible as a whole, but nothing in the sampling procedure tries to maximize how many individual positions are correct. The two goals are different, and so are the answers they produce.

If a model is graded per position, the best it can do is return the most likely amino acid at each position considered separately. Sampling does not do this. Greedy decoding does not either, because greedy chases the single most likely complete sequence, which is the right answer for a stricter metric that nobody reports.

This is a known result in other fields. It is called minimum Bayes risk decoding, it has been standard in machine translation since 2004, and it is the same distinction as Viterbi versus posterior decoding in hidden Markov models. I did not find it applied to protein sequence design. The derivation is in [`docs/theory.md`](docs/theory.md).

## The modification

I replaced the decoding step. Rather than committing to one random ordering and sampling a sequence from it, I run the model across many sampled orderings, collect the per-position probability distribution each run produces, average those distributions, and take the most likely amino acid at each position.

Averaging across orderings is what makes the estimate meaningful. A single run tells you each residue's probability conditioned on whatever happened to be placed before it, which is an accident of the ordering. Averaging over many orderings removes that dependence, so what remains approximates the probability of each amino acid at each position given the backbone alone.

```python
total = None
for i in range(n_samples):
    out = runner.sample(temperature=0.5, rng=np.random.default_rng(seed + i))
    probs = to_numpy(out["probs"])[0]      # (L, 21), normally discarded
    total = probs if total is None else total + probs

marginals = total / n_samples
sequence = marginals.argmax(axis=-1)       # most likely amino acid per position
```

This improved NSR by 1.51%, with a confidence interval of [+0.48%, +2.37%], and it was better than the baseline on 17 of the 19 chains. The per-position probabilities it uses were already being computed on every sampling run and discarded.

The implementation is in `experiments/run_consensus.py`.

### A cheaper route to most of the gain

While benchmarking this I checked whether ProteinMPNN already contained anything comparable, and it does. A function called `unconditional_probs` returns per-position probabilities in a single forward pass, computed without any sequence context. Taking its argmax improves NSR by 1.23%.

Compared directly, my version is better by 0.28%, with a confidence interval of [-0.84%, +1.44%]. That is not a distinguishable difference. So the modification I designed outperforms the default decoder and roughly matches a cheaper path that was already in the library but had never been connected to the metric.

Both findings support the same conclusion, which is that the default decoder optimizes the wrong objective and any estimator targeting per-position probabilities beats it.

## The configuration bug

The model was being constructed with the wrong graph size:

```python
model = ProteinMPNN(..., k_neighbors=64)
model.load_state_dict(torch.load("v_48_020.pt")["model_state_dict"])
```

The 48 in the checkpoint filename is the number of neighbours those weights were trained with. The `k_neighbors` argument controls graph topology rather than tensor shapes, so `load_state_dict` succeeds without an error or a warning and the model silently runs on a graph it has never seen.

Reading the value from the checkpoint recovers 2.04% NSR:

```python
checkpoint = torch.load(path, map_location=device)
k_neighbors = int(checkpoint.get("num_edges", 48))
model = ProteinMPNN(..., k_neighbors=k_neighbors)
```

## The modification that did not work

I started from the idea that revisiting the amino acids placed early, once the rest of the sequence is known, should improve the result. It does not.

My first implementation had a bug that meant the second pass never saw the draft sequence it was supposed to be revising. ProteinMPNN only reads a supplied sequence at positions marked as fixed, so passing a draft with every position unlocked discards it and resamples from scratch. The full account is in [`docs/findings.md`](docs/findings.md).

After fixing that, second-pass refinement still came out at -0.33%. Choosing which residues to revisit by the model's own confidence never beat choosing them at random, which means the model's uncertainty does not identify where it is wrong. The original hypothesis is wrong rather than badly implemented.

## How I measured it

The effects here are small relative to the noise, so most of the work was in measuring them properly.

Running ProteinMPNN twice on the same structure gives sequences that differ by around 4% NSR, purely from the random decoding order. That is several times larger than any effect I was looking for. Every modification therefore runs on the same seeds as the baseline, so that variance cancels in the comparison.

I treat the structure as the unit of analysis rather than the trial. Eight trials on one protein are not eight independent pieces of evidence that a method works. NSR is averaged within a structure first, then the per-structure differences are compared across structures.

Every modification that spends extra compute has a control that spends the same compute at random. Without one, redesigning a quarter of the residues and watching NSR rise looks like the selection method working when it is only the extra sampling.

I also found a bug in my own measurement code. Sampling drew from PyTorch's global random number generator, so my seeds fixed the decoding order and left the amino acid draws unseeded, and two runs with the same seed returned different sequences. I fixed it, added a regression test, and reran the affected numbers.

## Install

```bash
git clone https://github.com/<you>/proteinmpnn-decoding.git
cd proteinmpnn-decoding
pip install -e ".[model]"
./scripts/setup_proteinmpnn.sh
```

The library and the test suite need only NumPy. PyTorch is required to run the model itself.

## Running it

```bash
python experiments/run_estimator_comparison.py \
    --set benchmark_set.json \
    --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \
    --out results/estimators.csv

python experiments/analyze_experiments.py --csv results/estimators.csv
```

The algorithms can also be run with no weights and no PyTorch, against a small Potts model that implements the same interface:

```bash
python experiments/run_benchmark.py --toy --trials 20
```

## Repository layout

| Path | Contents |
|---|---|
| `src/ipr/` | Decoding, conditioning, uncertainty measures, metrics, model wrapper |
| `experiments/` | Benchmark scripts and the statistical analysis |
| `results/` | Raw CSV output behind every number in the documentation |
| `docs/` | Results, theory, findings, reproduction steps |
| `notes/` | The original Colab notebook and project notes |
| `tests/` | 51 tests, 48 of which run without PyTorch |

## Limitations

The 19 chains came from PDB files bundled inside a few Python packages, which makes this a pilot rather than a benchmark. Several are modified structures, chains taken from complexes are designed without their binding partners, and I tested only one checkpoint.

More importantly, marginal decoding maximizes per-position agreement and makes no promise that the resulting sequence is coherent as a whole, because it never asks whether the chosen amino acids fit together. That is the correct tradeoff for NSR. If the objective were foldability or stability, the metric would be different and so would the right decoder. I did not measure that.

## Documentation

[`docs/results.md`](docs/results.md) has every experiment with full tables. [`docs/theory.md`](docs/theory.md) derives why per-position probabilities are the right target and records which framings I tried and rejected. [`docs/findings.md`](docs/findings.md) covers four implementation bugs in the original work with citations to the ProteinMPNN source. [`docs/reproducing.md`](docs/reproducing.md) gives the exact commands.

## Citation

```bibtex
@article{dauparas2022proteinmpnn,
  title   = {Robust deep learning-based protein sequence design using ProteinMPNN},
  author  = {Dauparas, J. and Anishchenko, I. and Bennett, N. and others},
  journal = {Science},
  volume  = {378},
  number  = {6615},
  pages   = {49--56},
  year    = {2022},
  doi     = {10.1126/science.add2187}
}
```

## License

MIT. ProteinMPNN is MIT licensed and is not vendored in this repository.
