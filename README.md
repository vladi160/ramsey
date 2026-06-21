# Ramsey(5,5) — Computational Search: Code, Data, and Verification

Code and data accompanying the paper:

> **No Valid Ramsey(5,5;42) Coloring is Circulant on Z₄₂:  
> An Exhaustive Proof with Z₂₁-Bi-Circulant Search and 84-Violation Record**  
> Vladi Ivanov, 2026  
> Paper: [`papers/ramsey_circulant_v1.pdf`](papers/ramsey_circulant_v1.pdf)

---

## Main results at a glance

| Claim | Method | Runtime | Script |
|---|---|---|---|
| No Z₄₂-circulant is Ramsey(5,5;42) valid | Exhaustive, 1,048,575 sets | ~8 min | `ramsey_circulant.py` |
| No Z₄₃-circulant is Ramsey(5,5;43) valid | Same 1,048,575 sets | ~7 min | `ramsey_circulant.py` |
| 84 K₅ violations on K₄₂ (§7.4) | Z₂₁ bi-circulant tabu | <3s per run | `ramsey_symmetry.py` |
| 84v = vertex-transitive floor (§9) | D₂₁, Z₇×S₃, Z₇ all reach 84 | ~20 min each | `ramsey_cayley.py` |
| 129 K₅ violations on K₄₃ (§7.9) | Unconstrained Rust tabu, tenure 25 | ~3 h | `rust/ramsey/` |

---

## Reviewer verification (quick)

A single script verifies the three core claims independently:

```
python verify.py
```

**What it checks:**

| Check | What | Runtime |
|---|---|---|
| 1 | Stored colorings have claimed K₅ violation counts (independent K₅ counter) | ~5s total |
| 2 | Exhaustive circulant proof works: degree-20 slice (C(21,10)=352,716 sets), 0 valid found | ~10 min |
| 3 | Bi-circulant tabu reaches 84 violations within 1000 steps from a random start | ~6s |

Run individual checks:
```
python verify.py --check 1   # verify stored colorings  (~5s, no dependencies)
python verify.py --check 2   # circulant proof slice    (~10 min, no dependencies)
python verify.py --check 3   # reproduce 84v search     (~6s, requires numpy)
```

**Dependencies:**
- Python 3.8+, no third-party packages for checks 1 and 2
- `numpy` for check 3: `pip install numpy`

---

## Reproducing every result from the paper

### Theorem 1 & 2: Exhaustive circulant proof

```bash
# K_42: all 1,048,575 generating sets — ~8 min
python scripts/ramsey_circulant.py --n 42 --s 5

# K_43: same 1,048,575 candidates — ~7 min
python scripts/ramsey_circulant.py --n 43 --s 5
```

Expected output: `No valid Ramsey(5,5;N) circulant found. Checked 1048575 candidates.`

### §7.4: Z₂₁ bi-circulant tabu — 84-violation record

```bash
# Seed 42 — reaches 84 within ~500 steps
python scripts/ramsey_symmetry.py --mode bicirculant --steps 5000 --tabu 8 --seed 42

# Seed 99
python scripts/ramsey_symmetry.py --mode bicirculant --steps 5000 --tabu 8 --seed 99
```

Expected: `** NEW BEST: 84 violations` within 600 steps.

### §7.5: Half-identical (HI) and self-complementary (SC) subspaces

```bash
# HI: same 84v floor
python scripts/ramsey_symmetry.py --mode hi --steps 10000 --tabu 8 --seed 42

# SC: floor at 126v — different basin
python scripts/ramsey_symmetry.py --mode sc --steps 50000 --tabu 8 --seed 42
```

### §7.6: Population crossover (confirms 84v = global minimum)

```bash
python scripts/ramsey_symmetry.py --mode crossover --steps-per 5000 --random-crosses 50
```

Expected: all 77 candidates converge to 84v; none below 84.

### §7.7: Z₇ hexa-circulant search

```bash
# Seed 7 reaches 84v (structurally distinct from Z₂₁ result)
python scripts/ramsey_symmetry.py --mode z7 --steps 10000 --tabu 8 --seed 7
```

### §7.8: K₄₃ bi-circulant (floor 168v)

```bash
python scripts/ramsey_symmetry.py --mode k43 --steps 100000 --tabu 8 --seed 42
```

### §7.9: K₄₃ unconstrained Rust tabu (129v record)

```bash
# From 168v warm-start, tenure 25 — reaches 129v (requires Rust binary, see below)
```

### §9 (Discussion): Cayley graph search (D₂₁, Z₇×S₃, F₄₂)

```bash
# D_21 — reaches 84v
python scripts/ramsey_cayley.py --group d21 --steps 100000 --restarts 20 --tabu 8

# Z_7 x S_3 — reaches 84v
python scripts/ramsey_cayley.py --group z7s3 --steps 100000 --restarts 20 --tabu 8

# F_42 (Frobenius) — floor at 210v
python scripts/ramsey_cayley.py --group f42 --steps 100000 --restarts 20 --tabu 8
```

### §8: IRDME structural analysis

```bash
python scripts/ramsey_irdme.py --checkpoint datasets/RAMSEY_K42_S5_bicirculant_84v.json
python scripts/ramsey_bipartite_to_irdme.py --checkpoint datasets/RAMSEY_K42_S5_bicirculant_84v.json
python scripts/ramsey_orbits_to_irdme.py --checkpoint datasets/RAMSEY_K42_S5_bicirculant_84v.json
```

### Rust unconstrained tabu (fastest solver)

```bash
cd rust/ramsey
cargo build --release

# K_42 from random start
./target/release/ramsey --n 42 --s 5 --iter 200000 --restarts 200 --tabu 12 --plateau 20000

# K_43 from 168v warm-start (write the bi-circulant checkpoint first, then):
./target/release/ramsey --n 43 --s 5 --iter 500000 --restarts 200 --tabu 25 \
    --plateau 100000 --checkpoint datasets/RAMSEY_K43_S5_bicirculant_best.json
```

**Requirements:** Rust toolchain (`rustup.rs`). Build time: ~30s.

---

## Datasets

All stored colorings are JSON files with fields:
- `n`: graph size
- `s`: clique size (always 5)
- `violations`: total monochromatic K₅ count (red + blue)
- `assignment`: list of 0/1, length `n*(n-1)/2`, edges ordered `(i,j)` with `i<j` lexicographically

| File | n | violations | Description |
|---|---|---|---|
| `RAMSEY_K42_S5_bicirculant_84v.json` | 42 | 84 | Seed-42 coloring (near-self-complementary) |
| `RAMSEY_K42_S5_bicirculant_84v_s99.json` | 42 | 84 | Seed-99 coloring (half-identical, B=A) |
| `RAMSEY_K42_S5_bicirculant_84v_hi_s42.json` | 42 | 84 | HI seed-42 coloring |
| `RAMSEY_K43_S5_rust_snap_003_v129.json` | 43 | 129 | Best known K₄₃ coloring (tabu tenure 25) |

---

## Repository layout

```
scripts/
  ramsey_circulant.py     Exhaustive circulant proof (Theorem 1: K₄₂, Theorem 3: K₄₃)
  ramsey_symmetry.py      Z₂₁ bi-circulant / Z₇ / K₄₃ tabu search
  ramsey_cayley.py        Cayley graph search (D₂₁, Z₇×S₃, F₄₂)
  ramsey_sat.py           DIMACS CNF encoder for kissat
  ramsey_hitting_set.py   Topological analysis of violation complex
  ramsey_irdme.py         IRDME structural analysis of colorings
  ramsey_bipartite_to_irdme.py  Bipartite IRDME (vertices + orbit types)
  ramsey_orbits_to_irdme.py     Orbit-type IRDME
  ramsey_local.py         Python local search (reference implementation)
rust/ramsey/              Rust tabu solver (~4,500 iters/s on K₄₂)
datasets/                 Stored colorings (see table above)
papers/                   Paper PDF and LaTeX source
verify.py                 One-command reviewer verification
```

---

## Citation

```bibtex
@misc{ivanov2026ramsey,
  author = {Vladi Ivanov},
  title  = {No Valid {R}amsey(5,5;42) Coloring is Circulant on $\mathbb{Z}_{42}$:
             An Exhaustive Proof with $\mathbb{Z}_{21}$-Bi-Circulant Search
             and 84-Violation Record},
  year   = {2026},
  note   = {Preprint}
}
```
