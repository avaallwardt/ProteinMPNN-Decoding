# Findings

What my original experiments showed, and four implementation problems I found while reconstructing them.

Everything here can be checked. The raw numbers are in [`notes/notebook_results.csv`](../notes/notebook_results.csv), the aggregates come from `python experiments/analyze_notebook_results.py`, and the behavioural claims are held in place by `tests/test_conditioning.py`.

## 1. The second pass never conditioned on the draft

Every two-pass experiment I wrote built its refinement pass this way:

```python
chain_M_total_refresh = torch.ones_like(chain_M)   # every position = 1
out2 = model.sample(X, randn2, S_draft, chain_M_total_refresh, ...)
```

The intent was to hand the model `S_draft` so it could revisit its early choices knowing the whole sequence.

ProteinMPNN does something else with it. In the decoding loop of `ProteinMPNN.sample`:

```python
S_t = (S_t*chain_mask_gathered + S_true_gathered*(1.0-chain_mask_gathered)).long()
```

`S_true` contributes only where `chain_mask` is 0. Where it is 1, the freshly sampled amino acid wins and the draft value at that position is dropped.

The ordering comes from two lines earlier:

```python
decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn)))
```

Multiplying by `chain_mask` pushes fixed positions down to roughly 0.0001, so they sort to the front and are embedded into the decoder state before anything else is sampled. That is the mechanism by which fixed positions become context.

With `chain_mask` set to all ones there are no fixed positions, nothing is embedded as context, and the draft has no path to the output. The second pass is an independent sample at the refinement temperature.

So the comparison I actually ran was not one pass against two. It was one pass at T=0.1 against a discarded pass at T=0.5 followed by one pass at T=0.1. The same algorithm on both sides, with wasted compute on one of them.

The measurements agree with that reading:

| Experiment | Standard | Two-pass | Paired difference | t |
|---|---|---|---|---|
| Recovery, 10 trials (cell 42) | 51.98% | 52.50% | +0.52% | +0.30 |
| Recovery, 5 trials (cell 40) | 50.53% | 50.26% | -0.27% | -0.10 |
| NLL, 10 trials (cell 46) | 1.0115 | 1.0179 | -0.0064 nats | -0.89 |

Three independent measurements, all consistent with zero. That is what you get when both arms are the same algorithm.

The fix is to invert the mask relative to the intuition. To condition on the draft, set `chain_mask` to 0 at the positions you want the model to read, and 1 only at the positions you want it to reconsider:

```python
from ipr.conditioning import refinement_mask
chain_mask = refinement_mask(chain_M, redesign_positions=uncertain)
```

This is implemented as `ipr.refine.conditional_refine`. Two tests hold the behaviour in place. One checks that identical output comes back from completely different drafts when every position is redesigned, reproducing the bug. The other checks that the output diverges once some positions are held.

The hypothesis was untested rather than disproven. My notes identified something real about autoregressive decoding and the proposed fix was sound, it just had not been run. Everything in `ipr.refine` other than `total_refresh` is a first attempt at running it properly, and [`results.md`](results.md) has what happened when I did.

## 2. Three cells measured teacher-forced accuracy rather than design recovery

Cells 18, 19 and 20 compute what they call pass 1 recovery through the forward pass:

```python
log_probs1 = model(X, S_native, mask, chain_M, residue_idx, chain_encoding_all, randn)
S_pass1 = torch.argmax(log_probs1, dim=-1)
```

`forward` is teacher-forced. It embeds whatever sequence you pass as decoder context at every position, so passing `S_native` means each position is predicted with the native sequence already in context. The answer is in the input.

That is why those cells report 57.89% while the sampling path reports 47 to 53% on the same structure. The 57.89% is reachable only when the native sequence is already known, so it is not a design result and it should not sit in the same table as one.

`Runner.log_probs` carries a warning about this, and `Runner.sample` is the only path that produces a real design recovery.

## 3. The targeted polish unlocked every residue

Cell 35 picks residues by confidence threshold and prints:

```
Residues Unlocked for Editing: 76.0 out of 76.0
```

All 76 residues of ubiquitin were selected, so what was meant to be a targeted edit was a full redesign.

The cause is where the probabilities came from. The `probs` output of `sample` is multiplied by the chain mask:

```python
all_probs.scatter_(1, ..., (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
```

Every fixed position therefore reports a probability of exactly 0, which is absent confidence rather than low confidence. Any threshold above zero selects all of them. Cell 35 scored with `torch.zeros_like(chain_M)`, which fixes everything, so every position read as zero confidence and every position was unlocked.

Two guards exist now. `ipr.uncertainty.select_top_k` fixes the budget explicitly, so a selection cannot silently expand to the whole sequence, and `ipr.refine.gibbs_sweep` raises if batched mode is handed every position, since that arrangement conditions on nothing. To score positions that are being held fixed, use `Runner.log_probs`, which is not masked.

## 4. The model was built with the wrong neighbour count

The notebook constructs the model with `k_neighbors=64` and loads `v_48_020.pt`. The 48 in that filename is the neighbour count those weights were trained with.

`k_neighbors` does not change any parameter shapes, so `load_state_dict` succeeds without a warning. The model runs on a 64-neighbour graph it never saw in training. `ipr.backends.load_model` reads `checkpoint['num_edges']` instead.

Measured on 19 chains with paired trials:

| Arm | Recovery | Gain | 95% CI | t | Better on |
|---|---|---|---|---|---|
| k=48, from the checkpoint | 45.65% | | | | |
| k=64, the notebook configuration | 43.62% | -2.04% | [-3.01%, -1.17%] | -4.20 | 2 of 19 |

It also costs 0.11 nats of negative log likelihood, 0.9288 against 1.0395, the largest likelihood degradation of any arm I tested. Every recovery number collected in the original notebook was depressed by roughly two points because of this, and it turned out to be the single largest effect found anywhere in the project.

## What the original experiments could claim

On ubiquitin alone, 76 residues, with 10 paired trials, single-pass ProteinMPNN recovery was 51.98% with a standard deviation of 2.17% across decoding orders. No two-pass variant tested was distinguishable from it.

The largest single-trial number recorded, 56.58%, sits inside the baseline's own spread, whose best trial was 55.26%. Quoting a peak across trials as a method result overstates it, because the best of ten noisy trials beats the mean of ten noisy trials automatically.

The notebook does not support a 2% improvement claim. The closest figure is the +0.52% paired mean over 10 trials, with a standard error of 1.73%, an interval that comfortably includes zero. The +6.58% that appears in cell 28 is a single trial measured against a deliberately degraded T=0.5 first pass rather than against the T=0.1 baseline, and any method that starts hot and ends cold looks good in that comparison, including a method that does nothing at all.

## Where the real improvements came from

All four candidates below were eventually run on 19 chains. Full results are in [`results.md`](results.md).

Fixing the neighbour-count mismatch gives +2.04%, with a confidence interval of [+1.17%, +3.01%], winning on 17 of 19 chains. It is the largest effect found anywhere in the project.

Order marginalization, meaning averaging per-position distributions across decoding orders and taking the argmax, gives +1.51% with a confidence interval of [+0.48%, +2.37%], winning on 17 of 19 chains. This is the marginal-mode estimator, which is the right one for a per-position accuracy metric like recovery.

Best-of-N selection by model score gives +0.20% with a confidence interval spanning zero. It improves negative log likelihood from 0.9288 to 0.9055 and does nothing for recovery. This is different from the tournament cell, which kept the best sample by recovery against the native sequence. That is selecting on the answer, and it cannot be reported as a method result.

Corrected conditional refinement gives -0.33% with a confidence interval spanning zero, and it never beats a random-selection control at the same budget. The hypothesis does not hold up once the conditioning bug is fixed.

## Sources

The ProteinMPNN source is at <https://github.com/dauparas/ProteinMPNN>, and the quoted lines from `protein_mpnn_utils.py` were verified against the `main` branch. The paper is Dauparas et al., *Robust deep learning-based protein sequence design using ProteinMPNN*, Science 378:49-56 (2022), <https://www.science.org/doi/10.1126/science.add2187>.

My original experiments are in `notes/proteinmpnn_w_edits_3_24_26.ipynb`, run between 2026-03-24 and 2026-03-28, and the progress presentation is `notes/ProteinMPNN_Progress_Update.pdf` from 2026-04-08.
