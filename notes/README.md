# Source materials

The original project artifacts, preserved unedited. Nothing here has been
corrected or cleaned up. The analysis lives in
[`docs/findings.md`](../docs/findings.md), and these files are the record it
refers to.

| File | What it is |
|---|---|
| `proteinmpnn_w_edits_3_24_26.ipynb` | The Colab notebook. 55 cells; outputs intact, which is where every number in `notebook_results.csv` comes from. Run 2026-03-24 to 03-28. |
| `ProteinMPNN_Progress_Update.pdf` | Lab progress presentation, 2026-04-08. States the problem and names the three challenges. |
| `PROTEINMPNN_PROJECT_notes.pdf` | Working notes: the statistical-mechanics and information-theory framing that `docs/theory.md` responds to. |
| `notebook_results.csv` | Every per-trial number transcribed from the notebook's cell outputs, in long format. |

## The notebook's structure

Reading it in order, the experiments were:

| Cells | What was tried | Recorded outcome |
|---|---|---|
| 0–12 | Setup, baseline single pass on 1UBQ | 52.63% recovery |
| 18–20 | Confidence-thresholded second pass via `forward` | 0.00% to −3.95%; **native-leaked, see findings §2** |
| 22–24 | Same via `sample`, temperature variants | −2.63% to 0.00% |
| 26 | Redesign the first-decoded 50% | −1.32% |
| 28–29 | "Total refresh" and a triple pass | +6.58% single trial; −5.26% for triple |
| 31–33 | 10-draft consensus, then best-of-10 tournament | 0.00%; peak 56.58% |
| 35 | Intended targeted polish | Unlocked all 76 residues, **findings §3** |
| 38–42 | Head-to-head benchmarks, 5 and 10 trials | −0.26% and +0.53% mean |
| 45–46 | NLL and perplexity; path-independence test | Arms indistinguishable |
| 48–54 | Second structure (3ALZ complex); ablation | Several cells errored |

The arc is worth preserving as-is: it is a record of a hypothesis being tested
honestly and repeatedly failing to show an effect. The reason it failed turned
out to be an implementation detail rather than the hypothesis, which is only
visible with the outputs kept alongside the code.

## A note on the 3ALZ cells

Cells 48–54 attempt a second structure and several of them error out
(`tied_featurize() got an unexpected keyword argument 'chain_id_dict'`,
`NameError: chain_id_dict`, `NameError: analyze_ablation`). The ablation in cell
54 did produce ten trials of output, which is in `notebook_results.csv` as
`ablation_complex`.

Cell 53 sets `chain_encoding_all = chain_M.clone().long()` and reports 0
residues in chain A and 76 in chain B. The shape is 76, which is ubiquitin's
length, not the 528 of the intended complex. The 3ALZ structure was never
successfully loaded, so the `ablation_complex` rows are ubiquitin again, not a
second structure. They are kept under that name because that is what the
notebook called them, but they should not be read as independent replication.
