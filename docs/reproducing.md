# Reproducing the experiments

Everything here runs on CPU. No GPU needed. The full set takes about two hours.

## 1. Set up

```bash
pip install -e ".[model]"
./scripts/setup_proteinmpnn.sh
export PYTHONPATH="$PWD/ProteinMPNN:$PYTHONPATH"
```

The weights ship with the ProteinMPNN repository, so there is nothing else to
download. `v_48_020.pt` is the one used throughout.

## 2. Build the benchmark set

The sandbox these experiments ran in could not reach `files.rcsb.org`, so the
structures came from PDB files bundled inside Python packages. If you can reach
RCSB, use a proper test split instead. See the limits section of
[`results.md`](results.md).

```bash
git clone --depth 1 https://github.com/prody/ProDy.git prody_src
git clone --depth 1 https://github.com/biopython/biopython.git biopython_src
git clone --depth 1 https://github.com/openmm/pdbfixer.git pdbfixer_src

python experiments/build_benchmark_set.py \
    --search ProteinMPNN/inputs prody_src/prody/tests/datafiles \
             biopython_src/Tests pdbfixer_src \
    --min-length 45 --max-length 450 --max-chains 60 \
    --extract-dir benchmark_chains \
    --out benchmark_set.json
```

This filters to chains with complete backbone coordinates, no unknown residues
and no low-complexity sequences, drops duplicate sequences, and writes each
surviving chain to its own PDB file. ProteinMPNN's featurizer needs every chain
in a file accounted for, so extracting one chain per file is what makes a
monomer design task out of a complex.

Expect 19 chains from the sources above.

## 3. Run

Main sweep, about 47 minutes:

```bash
python experiments/run_experiments.py \
    --set benchmark_set.json \
    --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \
    --trials 8 --best-of 4 --rounds 2 --fraction 0.25 \
    --temperature 0.1 --temperatures 0.3 0.03 0.01 \
    --out results/experiments_19chains.csv
```

Consensus decoding, about 25 minutes:

```bash
python experiments/run_consensus.py \
    --set benchmark_set.json \
    --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \
    --n-samples-list 8 32 --reps 4 --icm-sweeps 0 --temperatures 0.5 \
    --out results/consensus_paired.csv
```

End to end, original pipeline against the corrected one:

```bash
python experiments/run_pipeline_comparison.py \
    --set benchmark_set.json \
    --weights ProteinMPNN/vanilla_model_weights/v_48_020.pt \
    --n-samples 32 --reps 3 \
    --out results/pipeline_comparison.csv
```

## 4. Analyse

```bash
python experiments/analyze_experiments.py --csv results/experiments_19chains.csv
```

Averages trials within a structure, takes per-structure paired differences
against the baseline arm, and bootstraps a 95% interval over structures.

## Notes on getting comparable numbers

**Seeds are shared across arms.** Every arm in a given trial runs from the same
seed. This is what makes the comparison paired, and it matters more than it
sounds: trial-to-trial spread from the random decoding order is around 4%,
several times the size of the effects being measured.

**Sampling draws through torch's global RNG.** `Runner.sample` derives a torch
seed from the NumPy generator it is handed, so a call is reproducible from that
generator alone. If you call `model.sample` directly rather than through
`Runner`, seed `torch.manual_seed` yourself or your runs will not repeat.

**Absolute recovery will not match published ProteinMPNN figures.** Chains
taken from complexes are designed without their partners, so interface residues
have no context. The paired comparisons are unaffected because every arm sees
the same structures.

**Expect the small effects to move between runs.** Anything under about half a
percent with an interval touching zero should be treated as unresolved rather
than measured.
