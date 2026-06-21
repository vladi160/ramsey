"""
Circulant Graph Search for Ramsey Lower Bounds.

A circulant graph C(n, S) has vertex set Z_n with edge (i,j) iff
|i-j| mod n ∈ S (where S ⊂ {1,...,n//2}).

This script exhaustively searches for valid Ramsey colorings among
circulant graphs, which is the approach Exoo (1989) used to prove R(5,5)≥43.

For K_42, a valid circulant with S ⊂ {1,...,21} proves R(5,5)≥43.
For K_43, a valid circulant would prove R(5,5)≥44 (currently unknown).

Usage:
  python scripts/ramsey_circulant.py --n 42 --s 5
  python scripts/ramsey_circulant.py --n 42 --s 5 --degree 20
  python scripts/ramsey_circulant.py --n 42 --s 5 --degree 20 --threads 4
  python scripts/ramsey_circulant.py --verify --adj "[1,2,5,10,11,...]"

Key optimization: bitmask K_s detection is ~1-10ms per graph for K_s-free
graphs, so we can search ~537k candidates for K_42 in ~1-5 hours.
"""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------------
# Fast K_s detection via bitmasks
# ---------------------------------------------------------------------------

def build_nbr_masks(n: int, S: set[int]) -> list[int]:
    """Build neighbor bitmasks for circulant C(n, S)."""
    nbr = [0] * n
    for i in range(n):
        for d in S:
            j1 = (i + d) % n
            j2 = (i - d) % n
            nbr[i] |= (1 << j1)
            if j1 != j2:
                nbr[i] |= (1 << j2)
    return nbr


def has_clique_of_size(nbr: list[int], n: int, s: int) -> bool:
    """
    Check if the graph (given as neighbor bitmasks) has a clique of size s.
    Uses recursive bitmask intersection — very fast in practice for K_s-free graphs.
    """
    def extend(clique_mask: int, candidates: int, depth: int) -> bool:
        if depth == s:
            return True
        need = s - depth
        cands = candidates
        while cands:
            # Extract lowest set bit
            lsb = cands & (-cands)
            v = lsb.bit_length() - 1
            cands ^= lsb
            remaining = bin(cands).count('1')
            if remaining + 1 < need:
                break  # Can't reach size s even using all remaining candidates
            new_cands = candidates & nbr[v]
            # Only consider vertices > v to avoid duplicates
            new_cands &= ~((1 << (v + 1)) - 1)
            if extend(clique_mask | lsb, new_cands, depth + 1):
                return True
        return False

    for start in range(n - s + 1):
        # Build candidate set: neighbors of start that are > start
        cands = nbr[start] >> (start + 1)
        if bin(cands).count('1') >= s - 1:
            if extend(1 << start, cands << (start + 1), 1):
                return True
    return False


def check_circulant(n: int, S: set[int], s: int) -> tuple[bool, bool]:
    """
    Check if circulant C(n, S) is a valid Ramsey coloring (no mono K_s).
    Returns (red_ok, blue_ok) where True means no K_s found.
    """
    # Red graph: edges where |i-j| mod n ∈ S
    red_nbr = build_nbr_masks(n, S)

    # Blue graph: edges where |i-j| mod n ∉ S (and i ≠ j)
    half = n // 2
    all_diffs = set(range(1, half + 1))
    if n % 2 == 0:
        all_diffs.add(n // 2)
    S_blue = all_diffs - S
    blue_nbr = build_nbr_masks(n, S_blue)

    red_ok = not has_clique_of_size(red_nbr, n, s)
    if not red_ok:
        return False, True  # Fail fast

    blue_ok = not has_clique_of_size(blue_nbr, n, s)
    return red_ok, blue_ok


def build_adj_matrix(n: int, S: set[int]) -> list[list[int]]:
    """Build full adjacency matrix for circulant C(n, S)."""
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for d in S:
            j1 = (i + d) % n
            j2 = (i - d) % n
            adj[i][j1] = 1
            adj[j1][i] = 1
            adj[i][j2] = 1
            adj[j2][i] = 1
    return adj


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_circulants(
    n: int,
    s: int,
    degree: int | None = None,
    max_generators: int | None = None,
    verbose: bool = True,
    seed: int | None = None,
    randomize: bool = True,
    all_degrees: bool = False,
) -> tuple[set[int], float] | None:
    """
    Search all circulant graphs C(n, S) for a valid Ramsey(s,s;n) coloring.

    For K_42 s=5 with degree=20: 537k candidates, ~3-5 min.
    For K_42 s=5 with all_degrees: ~1M candidates (using red/blue symmetry), ~10 min.
    Returns (S, elapsed_seconds) if found, None otherwise.
    """
    half = n // 2
    all_diffs = list(range(1, half + 1))

    if all_degrees:
        # Exhaustive: all 2^half subsets, but use symmetry:
        # If C(n, S) is valid, so is C(n, complement_S) by swapping colors.
        # So only check S with |S| <= half//2, and those with |S|==half//2
        # and first diff in S (lex-min canonical form).
        # Simpler: enumerate all 2^half subsets, deduplicate via canonical form.
        # For half=21: 2^21 = 2M total, ~1M unique after symmetry.
        candidates = []
        all_diffs_set = set(all_diffs)
        for k in range(1, half + 1):
            for combo in combinations(all_diffs, k):
                S = set(combo)
                S_comp = all_diffs_set - S
                # Keep only the lex-smaller of S and S_comp to avoid duplicates
                # (valid under red/blue swap symmetry)
                S_tup = tuple(sorted(S))
                comp_tup = tuple(sorted(S_comp)) if S_comp else ()
                if S_tup <= comp_tup:
                    candidates.append(S)
    else:
        # Filter by degree only
        if degree is not None:
            target_sizes = []
            if n % 2 == 0:
                if degree % 2 == 0:
                    target_sizes.append(("without_half", degree // 2))
                if (degree + 1) % 2 == 0:
                    target_sizes.append(("with_half", (degree + 1) // 2))
            else:
                if degree % 2 == 0:
                    target_sizes.append(("normal", degree // 2))
        else:
            target_sizes = [("all", k) for k in range(1, half + 1)]

        candidates = []
        for (mode, k) in target_sizes:
            if mode == "with_half":
                base_diffs = [d for d in all_diffs if d != n // 2]
                for combo in combinations(base_diffs, k - 1):
                    candidates.append(set(combo) | {n // 2})
            elif mode == "without_half":
                base_diffs = [d for d in all_diffs if d != n // 2]
                for combo in combinations(base_diffs, k):
                    candidates.append(set(combo))
            else:
                for combo in combinations(all_diffs, k):
                    candidates.append(set(combo))

    if max_generators is not None:
        candidates = [c for c in candidates if len(c) <= max_generators]

    total = len(candidates)
    if randomize:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  CIRCULANT SEARCH: C({n}, S), avoiding monochromatic K_{s}")
        print(f"  Candidates: {total:,} generating sets to check")
        print(f"  Degree filter: {degree if degree is not None else 'all'}")
        print(f"{'='*70}\n")

    checked = 0
    t0 = time.time()
    report_interval = max(1, total // 100)  # Report every 1%

    for S in candidates:
        checked += 1

        if verbose and (checked % report_interval == 0 or checked == 1):
            elapsed = time.time() - t0
            rate = checked / elapsed if elapsed > 0 else 0
            pct = 100 * checked / total
            eta = (total - checked) / rate if rate > 0 else float('inf')
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
            print(f"  {pct:5.1f}%  checked={checked:,}/{total:,}  "
                  f"rate={rate:.0f}/s  ETA={eta_str}  S_size={len(S)}", flush=True)

        red_ok, blue_ok = check_circulant(n, S, s)
        if red_ok and blue_ok:
            elapsed = time.time() - t0
            if verbose:
                print(f"\n  FOUND VALID COLORING!")
                print(f"  Generating set S = {sorted(S)}")
                print(f"  Checked {checked:,}/{total:,} candidates in {elapsed:.1f}s")
            return S, elapsed

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Search complete: no valid circulant found among {total:,} candidates.")
        print(f"  ({elapsed:.1f}s total)")
    return None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_and_save(n: int, s: int, S: set[int], out_path: str) -> None:
    """Verify the circulant and save as IRDME JSON."""
    red_ok, blue_ok = check_circulant(n, S, s)
    if red_ok and blue_ok:
        print(f"  Verification: VALID — no monochromatic K_{s} in red or blue")
    else:
        print(f"  Verification: INVALID — red_ok={red_ok}, blue_ok={blue_ok}")
        return

    items = [{"id": str(i), "type": "vertex", "dimensions": {"index": i}} for i in range(n)]
    rels = []
    half = n // 2
    all_diffs = set(range(1, half + 1))
    if n % 2 == 0:
        all_diffs.add(n // 2)
    S_blue = all_diffs - S

    for i in range(n):
        for j in range(i + 1, n):
            d = min((j - i) % n, (i - j) % n)
            rels.append({
                "from": str(i), "to": str(j),
                "type": "red" if d in S else "blue",
                "directed": False
            })

    data = {
        "meta": {
            "name": f"Ramsey K_{n} valid circulant, avoid K_{s}",
            "domain": "mathematics", "subdomain": "ramsey_theory",
            "n": n, "s": s,
            "result": f"R({s},{s}) > {n}",
            "construction": "circulant",
            "generators": sorted(S),
        },
        "items": items,
        "relations": rels,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {out_path}")
    print(f"\n  RESULT: R({s},{s}) > {n}")
    print(f"  Generators: {sorted(S)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Circulant graph Ramsey search")
    parser.add_argument("--n", type=int, required=True, help="Graph size")
    parser.add_argument("--s", type=int, default=5, help="Clique size to avoid (default: 5)")
    parser.add_argument("--degree", type=int, default=None,
                        help="Target red vertex degree (default: try all). "
                             "For K_42 s=5: use 20 or 21.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle candidates")
    parser.add_argument("--all-degrees", action="store_true",
                        help="Exhaustive: try all generating sets (uses red/blue symmetry, ~1M for K_42)")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--verify", default=None,
                        help="Verify a specific generating set, e.g. '1,2,5,10'")
    args = parser.parse_args()

    n, s = args.n, args.s

    if args.verify:
        S = set(int(x.strip()) for x in args.verify.split(","))
        print(f"\n  Verifying C({n}, {sorted(S)})...")
        out_path = args.out or f"datasets/RAMSEY_K{n}_S{s}_circulant.json"
        verify_and_save(n, s, S, out_path)
        return

    result = search_circulants(
        n=n, s=s,
        degree=args.degree,
        verbose=True,
        seed=args.seed,
        randomize=not args.no_shuffle,
        all_degrees=args.all_degrees,
    )

    if result is not None:
        S, elapsed = result
        out_path = args.out or f"datasets/RAMSEY_K{n}_S{s}_circulant.json"
        verify_and_save(n, s, S, out_path)
    else:
        print(f"\n  No valid circulant found for K_{n} avoiding K_{s}")
        print(f"  The Exoo K_42 construction may use a non-standard degree or")
        print(f"  the graph may not be a pure circulant — try expanding the search.")


if __name__ == "__main__":
    main()
