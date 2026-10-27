# Roadmap

The experiments that would settle whether refinement helps, in dependency
order. Each one is cheap, and each has a result that is worth reporting whether
it comes out positive or negative.

The ordering matters: experiment 1 determines whether 3–6 are worth running at
all.

---

## 1. Measure the entropy gap *(do this first)*

**Question.** Do positions decoded early actually carry more conditional
uncertainty than positions decoded late?

This is the premise the entire project rests on, and it has never been measured.
It needs no refinement machinery, just one sampling pass per trial, reading
`out['probs']`.

**Method.** For many random decoding orders on several structures, record each
position's conditional entropy against its decoding rank. Regress entropy on
rank.

**What each outcome means.**

- *Clear negative slope* → there is a real uncertainty budget at the start of
  the order, and it quantifies the ceiling on what refinement can recover.
  Proceed to experiment 3.
- *Flat* → early positions are no less certain than late ones, the premise is
  wrong, and the project should stop. **This is a publishable negative result**
  and would be worth more than another inconclusive benchmark.

**Secondary read.** The same data answers a sharper question: is the entropy at
rank $t$ driven by $t$, or by the residue's burial? If buried residues are
confident regardless of rank, the order matters less than the structure does.

---

## 2. Measure conditional inconsistency

**Question.** How much do ProteinMPNN's single-site conditionals disagree across
decoding orders?

[`docs/theory.md §2`](theory.md#the-consistency-problem) notes that the
conditionals are order-dependent, so they may not come from any single joint
distribution. The size of that inconsistency determines whether Gibbs-style
refinement has a target to converge to.

**Method.** For a fixed sequence and position $i$, compute
$p(s_i \mid S_{-i})$ under many random orders that place $i$ last. Report the
spread: total variation between the conditionals, and entropy of their mean.

**What it means.** Small spread licenses the pseudo-Gibbs framing and justifies
averaging conditionals over orders. Large spread means single-site updates are
chasing a moving target, and the right response is order-marginalized
conditionals (average over sampled $\pi$ before taking the argmax), not more
sweeps.

---

## 3. Benchmark corrected refinement

**Question.** Does conditioning on the draft improve recovery?

The experiment the project set out to run. `experiments/run_benchmark.py`
already implements it.

**Method.**

- **Structures:** at least 20, not one. Ubiquitin alone cannot distinguish a 2%
  effect from decoding-order noise. Use a standard held-out set (the ProteinMPNN
  paper's test split) so numbers are comparable to published figures.
- **Pairing:** same seeds across arms. This one isn't optional. The per-trial
  spread is roughly 4%, several times the effect size.
- **Arms:** `single_pass` (control), `total_refresh` (the original, expected to
  match the control), `entropy_guided`, `margin_guided`, `early_decoded`, and
  `random_guided` at matched budget.
- **Report:** paired mean difference with a confidence interval, per structure
  and pooled. Not a peak across trials.

**The control that matters.** `random_guided` at the same `fraction`. If guided
selection cannot beat random selection on the same number of residues, the
uncertainty signal is doing nothing and any gain is from extra sampling alone.

---

## 4. Sweep the refinement budget

**Question.** How many residues should be reconsidered, and for how many rounds?

**Method.** Grid over `fraction` ∈ {0.05, 0.1, 0.25, 0.5} and `n_rounds` ∈
{1, 2, 4, 8}, on the structures from experiment 3.

**What to watch for.** A gain that grows monotonically with `fraction` toward
1.0 is suspicious, because at `fraction = 1.0` the pass conditions on nothing. A
monotone trend to that limit suggests the measured effect is extra sampling
rather than refinement. An interior optimum is the signature of genuine
conditioning.

Also track composition (`ipr.metrics.composition`). Iterated greedy updates tend
to drift toward whichever residues the model finds generically favourable. A
recovery gain that comes with a collapsing amino-acid distribution is a sign of
mode-seeking rather than better design.

---

## 5. Decoding order as a controlled variable

**Question.** Does a structured decoding order beat a random one?

`randn_for_order` makes this exact rather than a seed lottery.

**Method.** Compare `random`, `n_to_c`, `core_first`, `surface_first`,
`confident_first`, `uncertain_first`, and `blocked_random` at matched
temperature and trials.

**Prediction, stated in advance.** If the compounding-error story is
right, `core_first` should beat `surface_first`. Buried residues are the most
constrained, so fixing them first gives the best context for everything else.
Writing the prediction down before running makes the result informative either
way.

---

## 6. Soft feedback between passes

**Question.** Does passing distributions instead of argmax tokens recover the
information the progress deck identified as lost?

The deck's second challenge: feeding discrete argmax values between iterations
throws away the full distribution, so later passes cannot see which residues
were close runners-up.

**Method.** ProteinMPNN embeds the sequence through `self.W_s`, an
`Embedding(21, hidden_dim)`. The soft version replaces a one-hot lookup with the
expected embedding, `probs @ W_s.weight`. This requires a small wrapper around
the decoder rather than changes to `sample`.

**Caveat.** The model was trained only on one-hot inputs. Expected embeddings
are off the training manifold, and the result may be worse for that reason
alone. Worth an interpolation parameter $\alpha$ between one-hot and expected
embedding, with $\alpha = 0$ recovering current behaviour exactly.

Not yet implemented.

---

## 7. Structural ensembles

**Question.** Are recovery gains robust to backbone perturbation, and does
designing against an ensemble help?

From the progress deck's third challenge, and the notes' interest in protein
breathing.

**Method.** `load_model(..., backbone_noise=σ)` exposes ProteinMPNN's own
coordinate noise. Sweep σ, measure recovery degradation, then design against an
ensemble by averaging conditionals across perturbed backbones.

**Why it matters beyond robustness.** This is the only setting in which the
mutual-information framing from the notes becomes well posed. With one fixed
backbone there is no distribution over structures, so $I(S; X)$ is undefined.
With an ensemble it is computable. See
[`docs/theory.md §4`](theory.md#what-is-not-well-defined).

---

## Standing requirements

Applying to every experiment above:

1. **Report paired statistics with intervals**, never a peak across trials. The
   best of 10 noisy trials beats the mean of 10 noisy trials by construction.
2. **More than one structure.** Every number currently in this repository comes
   from ubiquitin.
3. **Keep the matched-budget control in every comparison.** Extra sampling
   masquerades as method improvement with remarkable consistency.
4. **Write the prediction down first.** With this many arms and this much
   variance, something will look significant. Deciding in advance what would
   count as a positive result is the cheapest protection against reading noise.
