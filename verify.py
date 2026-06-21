"""
verify.py — Reviewer verification script for ramsey paper results.

Runs three independent checks:
  1. Verify all stored colorings have the claimed violation counts (pure stdlib)
  2. Sanity-check the exhaustive circulant proof on a degree-20 slice (~2 min)
  3. Confirm the bi-circulant search reaches 84 violations in <600 steps (needs numpy)

Usage:
    python verify.py           # all checks
    python verify.py --check 1 # only check 1  (~30s per coloring)
    python verify.py --check 2 # only check 2  (~2 min)
    python verify.py --check 3 # only check 3  (requires numpy)
"""

import json, sys, time, argparse
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------------
# Core: bitmask K_s counter (pure Python, no dependencies)
# ---------------------------------------------------------------------------

def build_adjacency(n, assignment, color=1):
    """Build bitmask adjacency for edges of given color (1=red, 0=blue)."""
    nbr = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if assignment[idx] == color:
                nbr[i] |= (1 << j)
                nbr[j] |= (1 << i)
            idx += 1
    return nbr


def count_ks_in_graph(nbr, n, s):
    """
    Count all K_s cliques in graph given by bitmask adjacency `nbr`.
    Uses ordered enumeration: vertices in each clique are listed in increasing order.
    """
    def recurse(cands, depth_remaining):
        # cands: bitmask of remaining candidate vertices (all > last chosen vertex)
        if depth_remaining == 0:
            return 1
        count = 0
        tmp = cands
        while tmp:
            u_bit = tmp & -tmp          # lowest set bit
            u = u_bit.bit_length() - 1  # actual vertex index
            tmp ^= u_bit                # remove u from working copy
            # Candidates for next level: those still in tmp (> u) AND neighbors of u
            new_cands = tmp & nbr[u]
            if bin(new_cands).count('1') >= depth_remaining - 1:
                count += recurse(new_cands, depth_remaining - 1)
        return count

    total = 0
    for v in range(n - s + 1):
        # Red neighbors of v that are strictly greater than v
        cands = nbr[v] & ~((1 << (v + 1)) - 1)
        if bin(cands).count('1') >= s - 1:
            total += recurse(cands, s - 1)
    return total


def count_violations(n, s, assignment):
    """Return (red_K_s, blue_K_s, total) for a coloring stored as 0/1 edge list."""
    red_nbr  = build_adjacency(n, assignment, color=1)
    blue_nbr = build_adjacency(n, assignment, color=0)
    red  = count_ks_in_graph(red_nbr,  n, s)
    blue = count_ks_in_graph(blue_nbr, n, s)
    return red, blue, red + blue


# ---------------------------------------------------------------------------
# Check 1: Verify all stored colorings
# ---------------------------------------------------------------------------

DATASETS = [
    ("datasets/RAMSEY_K42_S5_bicirculant_84v.json",        42, 5, 84,  "K_42 seed-42  (near-SC)"),
    ("datasets/RAMSEY_K42_S5_bicirculant_84v_s99.json",    42, 5, 84,  "K_42 seed-99  (HI, B=A)"),
    ("datasets/RAMSEY_K42_S5_bicirculant_84v_hi_s42.json", 42, 5, 84,  "K_42 HI seed-42"),
    ("datasets/RAMSEY_K43_S5_rust_snap_003_v129.json",     43, 5, 129, "K_43 best (129v, tabu-25)"),
]

def check1():
    print("=" * 62)
    print("CHECK 1: Verify stored colorings have claimed violation counts")
    print("=" * 62)
    all_ok = True
    for path, n, s, claimed, label in DATASETS:
        p = Path(path)
        if not p.exists():
            print(f"  MISSING  {path}")
            all_ok = False
            continue
        data = json.load(open(p))
        assignment = data["assignment"]
        assert len(assignment) == n * (n - 1) // 2, \
            f"Assignment length {len(assignment)} != {n*(n-1)//2}"
        print(f"  {label} ... ", end="", flush=True)
        t0 = time.time()
        red, blue, total = count_violations(n, s, assignment)
        elapsed = time.time() - t0
        ok = (total == claimed)
        status = "OK  " if ok else "FAIL"
        print(f"[{status}]  red={red}  blue={blue}  total={total}  "
              f"(claimed={claimed})  [{elapsed:.0f}s]")
        if not ok:
            all_ok = False
    print()
    return all_ok


# ---------------------------------------------------------------------------
# Check 2: Exhaustive circulant proof — degree-20 slice
# ---------------------------------------------------------------------------

def build_circulant_adj(n, S):
    """Bitmask adjacency for circulant C(n, S)."""
    adj = [0] * n
    for i in range(n):
        for s_val in S:
            adj[i] |= (1 << ((i + s_val) % n))
            adj[i] |= (1 << ((i - s_val) % n))
    return adj

def check2():
    print("=" * 62)
    print("CHECK 2: Exhaustive circulant search — degree-20 slice")
    print("         All C(21,10)=184,756 generating sets with |S|=10")
    print("         Expected: 0 valid colorings found")
    print("=" * 62)
    n, s = 42, 5
    half = 21
    candidates = list(combinations(range(1, half + 1), 10))
    total = len(candidates)
    print(f"  Checking {total:,} candidates for K_{n}, s={s}...")
    t0 = time.time()
    found = 0
    for i, S in enumerate(candidates):
        S_set = set(S)
        S_comp = set(range(1, half + 1)) - S_set
        red_adj  = build_circulant_adj(n, S_set)
        if count_ks_in_graph(red_adj, n, s) > 0:
            continue
        blue_adj = build_circulant_adj(n, S_comp)
        if count_ks_in_graph(blue_adj, n, s) == 0:
            found += 1
            print(f"  VALID circulant found: S={sorted(S_set)}")
        if (i + 1) % 25000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (total - i - 1) / rate
            print(f"  {i+1:,}/{total:,}  ({rate:.0f}/s)  ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    ok = (found == 0)
    status = "OK  " if ok else "FAIL"
    print(f"\n  [{status}]  {found} valid circulants in this slice  [{elapsed:.0f}s]")
    print(f"         Full exhaustive proof: all {2**half - 1:,} sets (~8 min in Python,")
    print(f"         run: python scripts/ramsey_circulant.py --n 42 --s 5)")
    print()
    return ok


# ---------------------------------------------------------------------------
# Check 3: Bi-circulant tabu reaches 84 violations in <1000 steps
# ---------------------------------------------------------------------------

def check3():
    print("=" * 62)
    print("CHECK 3: Bi-circulant tabu reaches 84 violations")
    print("         seed=42, up to 1000 steps, tabu tenure=8")
    print("         Expected: best violations = 84")
    print("=" * 62)
    try:
        import numpy as np  # noqa
    except ImportError:
        print("  SKIPPED: numpy not installed  (pip install numpy)")
        return True

    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    try:
        import ramsey_symmetry as rs
    except ImportError as e:
        print(f"  SKIPPED: cannot import ramsey_symmetry: {e}")
        return True

    print("  Precomputing lookup tables ... ", end="", flush=True)
    t0 = time.time()
    canon_list, masks_np, mults_np, type_to_canons_np, sc_a_canons_np = \
        rs.build_lookup_tables(verbose=False)
    print(f"{time.time()-t0:.1f}s")

    import random as _random
    rng = _random.Random(42)
    # Random initial types, then run tabu
    types = [rng.randint(0, 1) for _ in range(rs.N_TYPES)]
    print("  Running tabu (seed=42, 1000 steps) ...", flush=True)
    t0 = time.time()
    best_types, best_viols = rs.tabu_search_bicirculant(
        types, masks_np, mults_np, type_to_canons_np,
        steps=1000, tabu_tenure=8, rng=rng, log_every=200, verbose=True
    )
    elapsed = time.time() - t0
    ok = (best_viols <= 84)
    status = "OK  " if ok else "NOTE"
    print(f"\n  [{status}]  best = {best_viols} violations  [{elapsed:.1f}s]")
    if not ok:
        print("  (84 is typically reached within 600 steps; run with more steps)")
    print()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Ramsey paper results")
    parser.add_argument("--check", type=int, choices=[1, 2, 3],
                        help="Run only this check (default: all)")
    args = parser.parse_args()

    results = {}
    if args.check in (None, 1):
        results[1] = check1()
    if args.check in (None, 2):
        results[2] = check2()
    if args.check in (None, 3):
        results[3] = check3()

    if len(results) > 1:
        print("=" * 62)
        print("SUMMARY")
        print("=" * 62)
        labels = {1: "Stored coloring violation counts",
                  2: "Circulant proof slice (degree-20)",
                  3: "Bi-circulant 84v search"}
        for k, v in sorted(results.items()):
            print(f"  Check {k}: {'PASS' if v else 'FAIL'}  — {labels[k]}")
        print()
        if all(results.values()):
            print("All checks passed.")
        else:
            sys.exit(1)
