# Theory

Why ProteinMPNN's decoding procedure is mismatched to the metric it is scored on, and what follows from fixing that.

The short version is this. ProteinMPNN is evaluated on native sequence recovery, which scores every position separately. The rule that minimizes expected error under a per-position score is to take the most likely amino acid at each position on its own. Sampling a sequence does not do this, and neither does greedy decoding. These are three different estimators, they disagree, and the gap between them is worth about a point and a half of recovery.

This is minimum Bayes risk decoding. It is standard in other fields and largely absent from protein design.

## 1. Setup

ProteinMPNN factorizes the sequence distribution over a decoding order, a permutation $\pi$ of the $L$ positions:

$$p_\pi(S \mid X) = \prod_{t=1}^{L} p\big(s_{\pi(t)} \mid s_{\pi(<t)}, X\big)$$

The backbone $X$ is fixed throughout. The sequence space is $\mathcal{S} = \mathcal{A}^L$ with $|\mathcal{A}| = 21$.

One detail matters later. ProteinMPNN does not define a single distribution. It defines a different one for each decoding order, and it draws $\pi$ at random for every sample. The object actually being sampled from is a mixture over orderings:

$$p(S \mid X) = \mathbb{E}_{\pi}\big[\, p_\pi(S \mid X) \,\big]$$

## 2. Which estimator the metric asks for

Designing a sequence is an estimation problem. You have a distribution over sequences and you must return one of them. Which one you should return depends entirely on how you will be scored.

Given a loss $\ell(\hat{S}, S)$ between your answer and the truth, the estimator minimizing expected loss is

$$\hat{S}_{\text{MBR}} = \arg\min_{\hat{S}} \; \mathbb{E}_{S \sim p(\cdot \mid X)}\big[\ell(\hat{S}, S)\big]$$

This is the minimum Bayes risk estimator. The important part is that it is a function of the loss. Change the loss and the optimal answer changes, even though the model has not.

### Recovery is Hamming loss

Recovery is the fraction of positions matching the native sequence. As a loss, that is Hamming distance normalized by length:

$$\ell(\hat{S}, S) = \frac{1}{L}\sum_{i=1}^{L} \mathbb{1}\big[\hat{s}_i \neq s_i\big]$$

Hamming loss decomposes over positions. Each position contributes independently and no term couples $\hat{s}_i$ to $\hat{s}_j$, so the expectation decomposes as well:

$$\mathbb{E}_S\big[\ell(\hat{S}, S)\big] = \frac{1}{L}\sum_{i=1}^{L} \Pr\big[\hat{s}_i \neq s_i\big] = \frac{1}{L}\sum_{i=1}^{L} \big(1 - p(s_i = \hat{s}_i \mid X)\big)$$

Every term is minimized separately, which gives

$$\hat{s}_i = \arg\max_{a \in \mathcal{A}} \; p(s_i = a \mid X)$$

the per-position marginal mode. Under Hamming loss that is the optimal estimator, and it needs the marginals rather than the joint.

### What the standard procedure returns instead

Ancestral sampling returns a draw from $p_\pi(S \mid X)$ for one random $\pi$. That is not an estimator of anything in particular, it is a sample. Its expected Hamming loss is strictly worse than the marginal mode's whenever the model has any uncertainty at all.

Greedy decoding returns approximately $\arg\max_S p_\pi(S \mid X)$, the joint mode. That is the minimum Bayes risk estimator for 0-1 loss on the entire sequence, which scores zero unless every position is correct. Nobody reports that metric.

So there are three different things here:

| Estimator | Optimal for | What it returns |
|---|---|---|
| Ancestral sample | nothing in particular | a draw from the joint |
| Joint mode, or greedy | 0-1 loss on the entire sequence | the single most likely sequence |
| Marginal mode | Hamming loss, which is recovery | the most likely amino acid at each position |

Recovery is what gets reported, and the estimator matched to it is the one nobody uses.

## 3. This is old mathematics in a new place

The same distinction appears wherever sequence models meet per-token metrics.

In hidden Markov models, Viterbi decoding returns the most likely state path and posterior decoding returns the most likely state at each position. They routinely disagree, and the textbook advice is to use posterior decoding when you are scored per position. Durbin et al. covers this directly.

In machine translation, minimum Bayes risk decoding has been standard since Kumar and Byrne in 2004, precisely because BLEU is not 0-1 loss on the whole output.

For neural sequence models generally, Eikema and Aziz argue that MAP decoding is the wrong default for a model whose probability mass is spread thin, and that sampling followed by minimum Bayes risk is better behaved.

The contribution here is noticing that protein design inherited ancestral sampling as its default, reports a per-position metric, and that nobody closed the loop between the two. That is a real gap and it is worth about 1.5% recovery.

## 4. Two ways to estimate the marginals

The marginal $p(s_i \mid X)$ requires summing over every other position, which is intractable. Two approximations are available and they turn out to be quite different.

### The context-free marginal, already in the library

ProteinMPNN ships `unconditional_probs`, which zeroes the decoder's backward attention mask so that every position is predicted from structure alone:

```python
order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
```

One forward pass, deterministic, no sampling. It is a marginal estimate that has been in the library since release.

Its weakness is that it throws away sequence context entirely. It cannot use the fact that a particular amino acid at position $j$ makes a particular amino acid at position $i$ more likely, because it never sees position $j$.

### Averaging over decoding orders

Every ancestral sample already computes, at each position, the conditional

$$p\big(s_i \mid s_{\pi(<i)}, X\big)$$

under its own ordering and prefix. Those are returned in the `probs` array and normally discarded. Averaging them across $N$ sampled orderings gives

$$\hat{p}(s_i \mid X) = \frac{1}{N}\sum_{k=1}^{N} p\big(s_i \mid s^{(k)}_{\pi_k(<i)}, X\big)$$

which integrates over both the prefix and the ordering. Unlike the context-free version, every contributing distribution carries real sequence context. The cost is $N$ forward passes, though they are passes you were already paying for if you sample more than once.

### Which is better is an empirical question

Both target $p(s_i \mid X)$. The direct one asks the model for exactly that quantity but answers from structure alone. The sampled one sees context but inherits whatever bias the model's own sampling has.

I measured them against each other in `experiments/run_estimator_comparison.py`. Both beat sampling. The order-averaged version wins by 0.28% with a confidence interval of [-0.84%, +1.44%], which is not a distinguishable difference. Full numbers in [`results.md`](results.md).

## 5. A wrinkle specific to ProteinMPNN

One thing here does not appear in the HMM or translation versions of the problem.

Because the model defines a different joint $p_\pi$ for each ordering, the conditionals $p(s_i \mid s_{-i})$ obtained under different orderings need not agree, and need not be the conditionals of any single joint distribution. Three consequences follow.

Single-site update schemes are pseudo-Gibbs rather than Gibbs. The usual convergence guarantees do not transfer, because they assume the conditionals come from one joint. Iterated conditional modes loses its monotone descent property for the same reason, since there is no single energy being descended. And averaging over orderings, which Section 4 does for estimation, happens to smooth this inconsistency as a side effect rather than as its motivation.

This is a known issue for masked and autoregressive language models used as Markov random fields. Besag's work on pseudo-likelihood sets out the conditions under which a set of conditionals corresponds to a joint, and they are not satisfied here.

## 6. Framings I tried and rejected

I am keeping these because knowing why they fail is worth more than quietly dropping them.

### Contraction mapping

My original plan was to show that the refinement step is a contraction and then invoke Banach's fixed point theorem to guarantee convergence to a unique optimum.

This cannot work, for a reason that has nothing to do with proteins. A contraction requires $d(T(x), T(y)) \le L \, d(x,y)$ with $L < 1$ for all pairs. Sequence space under Hamming distance is discrete, so its minimum nonzero distance is 1. If $d(x,y) = 1$ and $T(x) \neq T(y)$, then $d(T(x),T(y)) \ge 1 = d(x,y)$, giving $L \ge 1$. No nonconstant map on a discrete metric space is a contraction, so Banach does not apply here regardless of how the proof is attempted.

Two things survive from the idea. Iterating and measuring how many positions change per sweep gives an empirical convergence rate, which `ipr.refine.contraction_factor` reports with the caveat that one trajectory is not a supremum. And relaxing to distributions rather than amino acids puts the map on the simplex, where a Lipschitz constant is well defined and could be bounded through the model's Jacobian. That second route is where the instinct about analyzing the Jacobian actually pays off, and it connects to the soft-feedback idea from the progress deck.

### Mutual information with the fold

Asking how many bits the second pass adds about the fold requires a distribution over folds. There is one backbone here, held fixed, so there is no entropy to reduce and the quantity is undefined as stated.

It becomes well posed with a structural ensemble, which is what the protein-breathing idea supplies. Perturb the backbone and $I(S; X)$ over that ensemble is computable. That is a real experiment rather than a rephrasing.

### KL divergence reduction

$D_{KL}(q^{(k)} \| p)$ needs both arguments named. If $p$ is the model's own joint then the sampler already targets it and any reduction is tautological. If $p$ is the distribution of natural sequences compatible with the fold then it is unknown and cannot be sampled. The usable proxy is held-out negative log likelihood under the model, which is what `Runner.score` reports.

That proxy turned out to be misleading in its own way. The refinement arm posted the second-best likelihood of any modification while scoring worse on recovery. Likelihood and recovery are not interchangeable, which is itself a reason to be careful about which objective you optimize.

## 7. Status of each claim

| Claim | Status |
|---|---|
| Recovery is Hamming loss, so its MBR estimator is the marginal mode | Proved, Section 2 |
| Ancestral sampling does not target that estimator | Proved, it targets no estimator |
| Marginal decoding beats sampling on recovery | Measured, +1.51%, see results.md |
| Averaging over orderings estimates the marginal | Proved as Monte Carlo, bias depends on the model |
| Early-decoded residues have less context | True by construction |
| A second pass that discards the draft helps | False, see findings.md |
| Conditioning on a draft is implementable | True, `ipr.conditioning` |
| Conditional refinement improves recovery | False, -0.33% on 19 chains |
| Entropy identifies refinable positions | False, never beats random selection |
| Single-site updates form a Gibbs sampler | False as stated, conditionals are order-dependent |
| Refinement is a contraction on sequence space | False, discrete spaces admit no contraction |
| Mutual information with the fold is quantifiable | Ill posed without a structural ensemble |

## References

Kumar and Byrne (2004), *Minimum Bayes-Risk Decoding for Statistical Machine Translation*, HLT-NAACL. The canonical statement of minimum Bayes risk for sequence models.

Eikema and Aziz (2020), *Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation*, COLING. Why the mode is a poor default and sampling-based minimum Bayes risk is better behaved.

Durbin, Eddy, Krogh and Mitchison (1998), *Biological Sequence Analysis*. Posterior decoding against Viterbi, the same distinction in the hidden Markov model setting.

Besag (1975), *Statistical Analysis of Non-Lattice Data*. Pseudo-likelihood, and when a set of conditionals corresponds to a joint distribution.

Besag (1986), *On the Statistical Analysis of Dirty Pictures*. Iterated conditional modes and its monotone descent property, which needs that correspondence.

Dauparas et al. (2022), *Robust deep learning-based protein sequence design using ProteinMPNN*, Science 378:49-56.

Granas and Dugundji (2003), *Fixed Point Theory*. Banach's theorem and the hypotheses it requires.
