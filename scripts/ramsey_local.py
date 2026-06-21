"""
Optimized local search for Ramsey lower bounds.

Key optimization over ramsey_attack.py: precomputes all C(n,s) cliques and
maintains an incremental violation count. Each flip updates only the ~9,880
cliques containing that edge, rather than re-scanning all 850k cliques.

Result: ~10-100x faster per iteration than the naive approach.

Usage:
  python scripts/ramsey_local.py --n 42 --s 5 --restarts 500 --iter 5000
  python scripts/ramsey_local.py --n 43 --s 5 --restarts 1000 --iter 5000
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Precomputed clique structure
# ---------------------------------------------------------------------------

def build_clique_structure(n: int, s: int):
    """
    Precompute:
    - edges: list of (i,j) pairs, i<j, 0-indexed
    - edge_idx: dict (i,j) -> k
    - cliques: list of lists of edge indices (each clique = C(s,2) edge indices)
    - e2c: edge_idx -> list of clique indices it appears in

    For K_42 s=5: 850k cliques, each edge in 9,880 cliques.
    Memory: ~850k × 10 × 4 bytes = ~34 MB. Fine.
    """
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_idx = {e: k for k, e in enumerate(edges)}
    num_edges = len(edges)

    print(f"  Building clique structure: n={n}, s={s}")
    print(f"  Edges: {num_edges}")

    cliques = []
    e2c = [[] for _ in range(num_edges)]  # edge_idx -> clique indices

    t0 = time.time()
    for ci, clique_verts in enumerate(combinations(range(n), s)):
        clique_edges = []
        for a in range(s):
            for b in range(a + 1, s):
                ek = edge_idx[(clique_verts[a], clique_verts[b])]
                clique_edges.append(ek)
                e2c[ek].append(ci)
        cliques.append(clique_edges)
        if ci % 100_000 == 0 and ci > 0:
            print(f"    cliques built: {ci:,} ({time.time()-t0:.1f}s)")

    print(f"  Total cliques: {len(cliques):,} (built in {time.time()-t0:.1f}s)")
    return edges, edge_idx, cliques, e2c


def init_counts(assignment: list[int], cliques: list[list[int]], num_cliques: int, clique_size_edges: int):
    """
    Initialize clique_red[ci] and clique_blue[ci] from current edge assignment.
    assignment[k] = 1 (red) or 0 (blue) for each edge k.
    Returns (clique_red, clique_blue, violation_set).
    violation_set contains clique indices ci where clique_red[ci]==clique_size_edges
    or clique_blue[ci]==clique_size_edges.
    """
    clique_red = [0] * num_cliques
    clique_blue = [0] * num_cliques
    violation_set = set()

    for ci, cedges in enumerate(cliques):
        for ek in cedges:
            if assignment[ek] == 1:
                clique_red[ci] += 1
            else:
                clique_blue[ci] += 1
        if clique_red[ci] == clique_size_edges or clique_blue[ci] == clique_size_edges:
            violation_set.add(ci)

    return clique_red, clique_blue, violation_set


# ---------------------------------------------------------------------------
# Optimized local search
# ---------------------------------------------------------------------------

def local_search(
    n: int,
    s: int,
    edges: list[tuple[int, int]],
    cliques: list[list[int]],
    e2c: list[list[int]],
    initial_assignment: list[int],
    max_iter: int = 5000,
    tabu_tenure: int = 10,
    noise_rate: float = 0.15,
    rng: random.Random | None = None,
    verbose: bool = True,
    report_every: int = 500,
) -> tuple[list[int] | None, int, int]:
    """
    Tabu search with ILS perturbation.
    Each step:
      1. Pick a random violated clique
      2. Score each of its C(s,2) edges: delta = net reduction in violations if flipped
      3. Choose best non-tabu edge (aspiration: allow tabu if it beats best overall)
      4. Flip it, update clique counts + violation set incrementally

    Returns (assignment_if_solved, iterations, best_violations).
    """
    if rng is None:
        rng = random.Random()

    clique_size_edges = s * (s - 1) // 2  # = 10 for s=5
    num_cliques = len(cliques)
    num_edges = len(edges)

    assignment = initial_assignment[:]
    clique_red, clique_blue, violation_set = init_counts(
        assignment, cliques, num_cliques, clique_size_edges
    )

    best_assignment = assignment[:]
    best_violations = len(violation_set)
    stuck_since = 0
    tabu: dict[int, int] = {}  # edge_k -> iteration when tabu expires

    for it in range(max_iter):
        v_count = len(violation_set)

        if v_count < best_violations:
            best_violations = v_count
            best_assignment = assignment[:]
            stuck_since = it

        if v_count == 0:
            if verbose:
                print(f"  [iter {it}] SOLVED! Valid coloring found.")
            return assignment, it, 0

        if verbose and it % report_every == 0:
            print(f"  iter {it:>7}  violations={v_count:>5}  best={best_violations:>5}  "
                  f"tabu={len(tabu)}", flush=True)

        # ILS: perturb when stuck — escalating perturbation size
        stuck_thresh = max(500, tabu_tenure * 15)
        stuck_duration = it - stuck_since
        if stuck_duration >= stuck_thresh:
            assignment = best_assignment[:]
            clique_red, clique_blue, violation_set = init_counts(
                assignment, cliques, num_cliques, clique_size_edges
            )
            # Escalating perturbation: the longer we're stuck, the bigger the kick
            n_kicks = stuck_duration // stuck_thresh  # 1, 2, 3... with repeated stagnation
            perturb_frac = min(0.30, 0.08 * n_kicks)  # 8%, 16%, 24%, max 30%
            perturb = max(5, int(num_edges * perturb_frac))
            for _ in range(perturb):
                ek = rng.randrange(num_edges)
                _flip_edge(ek, assignment, clique_red, clique_blue, violation_set,
                           e2c, cliques, clique_size_edges)
            tabu.clear()
            stuck_since = it
            if verbose and it % report_every < stuck_thresh:
                print(f"  [ILS perturb at iter {it}: {perturb} edges ({perturb_frac*100:.0f}%)]")
            continue

        # Pick a random violated clique
        ci = rng.choice(list(violation_set))
        clique_edges = cliques[ci]

        # Noise: occasionally pick a random edge instead of best
        if rng.random() < noise_rate:
            ek = rng.choice(clique_edges)
            _flip_edge(ek, assignment, clique_red, clique_blue, violation_set,
                       e2c, cliques, clique_size_edges)
            tabu[ek] = it + tabu_tenure
            if it % 1000 == 0:
                tabu = {k: v for k, v in tabu.items() if v > it}
            continue

        # Score each edge in the violated clique
        best_score = None
        best_ek = None

        for ek in clique_edges:
            is_tabu = tabu.get(ek, 0) > it

            # Compute delta: if we flip this edge, how many violations change?
            # delta = (violations_gained) - (violations_fixed)
            # violations_fixed = cliques containing ek that are currently violated
            #                    AND have this edge as the "offender"
            delta = _compute_delta(ek, assignment, clique_red, clique_blue,
                                   violation_set, e2c, cliques, clique_size_edges)

            # Aspiration: allow tabu if it would beat global best
            projected = v_count - delta
            allow = not is_tabu or (projected < best_violations)

            if not allow:
                continue

            score = delta  # higher = better (fixes more violations)
            if best_score is None or score > best_score:
                best_score = score
                best_ek = ek

        if best_ek is None:
            # All tabu: pick random from clique
            best_ek = rng.choice(clique_edges)

        _flip_edge(best_ek, assignment, clique_red, clique_blue, violation_set,
                   e2c, cliques, clique_size_edges)
        tabu[best_ek] = it + tabu_tenure

        if it % 1000 == 0:
            tabu = {k: v for k, v in tabu.items() if v > it}

    return None, max_iter, best_violations


def _compute_delta(
    ek: int,
    assignment: list[int],
    clique_red: list[int],
    clique_blue: list[int],
    violation_set: set,
    e2c: list[list[int]],
    cliques: list[list[int]],
    clique_size_edges: int,
) -> int:
    """
    If we flip edge ek, how many net violations are fixed (positive = better)?
    Fixed = violated cliques that become OK after flip.
    New = OK cliques that become violated after flip.
    Returns fixed - new.
    """
    is_red = assignment[ek] == 1
    fixed = 0
    new = 0
    for ci in e2c[ek]:
        rc = clique_red[ci]
        bc = clique_blue[ci]
        if is_red:
            # Flip: rc-1, bc+1
            rc2, bc2 = rc - 1, bc + 1
        else:
            # Flip: rc+1, bc-1
            rc2, bc2 = rc + 1, bc - 1
        was_violated = (rc == clique_size_edges or bc == clique_size_edges)
        now_violated = (rc2 == clique_size_edges or bc2 == clique_size_edges)
        if was_violated and not now_violated:
            fixed += 1
        elif not was_violated and now_violated:
            new += 1
    return fixed - new


def _flip_edge(
    ek: int,
    assignment: list[int],
    clique_red: list[int],
    clique_blue: list[int],
    violation_set: set,
    e2c: list[list[int]],
    cliques: list[list[int]],
    clique_size_edges: int,
) -> None:
    """Flip edge ek and update all incremental state."""
    is_red = assignment[ek] == 1
    assignment[ek] = 0 if is_red else 1
    for ci in e2c[ek]:
        old_rc, old_bc = clique_red[ci], clique_blue[ci]
        if is_red:
            clique_red[ci] -= 1
            clique_blue[ci] += 1
        else:
            clique_red[ci] += 1
            clique_blue[ci] -= 1
        new_rc, new_bc = clique_red[ci], clique_blue[ci]
        # Update violation set
        was_v = (old_rc == clique_size_edges or old_bc == clique_size_edges)
        now_v = (new_rc == clique_size_edges or new_bc == clique_size_edges)
        if was_v and not now_v:
            violation_set.discard(ci)
        elif not was_v and now_v:
            violation_set.add(ci)


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def random_seed(n: int, s: int, rng: random.Random) -> list[int]:
    """Random balanced coloring."""
    num_edges = n * (n - 1) // 2
    return [rng.randint(0, 1) for _ in range(num_edges)]


def greedy_seed(n: int, s: int, rng: random.Random, edges: list[tuple[int, int]],
                cliques: list[list[int]], e2c: list[list[int]]) -> list[int]:
    """
    Greedy seed: assign each edge the color that creates fewer new violations.
    """
    clique_size_edges = s * (s - 1) // 2
    num_edges = len(edges)
    assignment = [-1] * num_edges
    clique_red = [0] * len(cliques)
    clique_blue = [0] * len(cliques)

    order = list(range(num_edges))
    rng.shuffle(order)

    for ek in order:
        best_color = 0
        best_new_v = float('inf')
        for color in [0, 1]:
            new_v = 0
            for ci in e2c[ek]:
                if color == 1:
                    rc, bc = clique_red[ci] + 1, clique_blue[ci]
                else:
                    rc, bc = clique_red[ci], clique_blue[ci] + 1
                if rc == clique_size_edges or bc == clique_size_edges:
                    new_v += 1
            if new_v < best_new_v:
                best_new_v = new_v
                best_color = color

        assignment[ek] = best_color
        for ci in e2c[ek]:
            if best_color == 1:
                clique_red[ci] += 1
            else:
                clique_blue[ci] += 1

    return assignment


def _paley_red_edges(p: int) -> set[tuple[int, int]]:
    """
    Returns the set of red (i,j) pairs i<j for the Paley(p) coloring.
    Edge (i,j) is red iff (j-i) mod p is a quadratic residue mod p.
    Works for prime p ≡ 1 (mod 4). Verified valid for R(4,4;17).
    """
    qr = set()
    for x in range(1, p):
        qr.add((x * x) % p)
    red = set()
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in qr:
                red.add((i, j))
    return red


def structured_seed(n: int, s: int, rng: random.Random,
                    edges: list[tuple[int, int]], edge_idx: dict,
                    cliques: list[list[int]], e2c: list[list[int]]) -> list[int]:
    """
    FPL-guided structured seed: Paley(17) base + hub-balance greedy extension.

    Phase 1: Fix the first 17 vertices with Paley(17) — a valid K₄-free (hence
    K₅-free) Ramsey(4,4;17) coloring. This guarantees zero violations among all
    C(17,5)=6188 cliques contained in that subgraph.

    Phase 2: Extend the remaining 725 edges (involving vertices 17..n-1) using
    greedy assignment with FPL hub-balance as tiebreak:
    - Primary: pick the color creating fewer new violations (same as greedy_seed)
    - Tiebreak: pick the color keeping each vertex's red degree closer to (n-1)/2,
      enforcing the anti-correlation principle that valid Ramsey colorings require.

    The Paley base eliminates violations in a guaranteed violation-free subgraph,
    reducing overall starting violations vs. pure random-order greedy.
    """
    clique_size_edges = s * (s - 1) // 2
    num_edges = len(edges)
    assignment = [-1] * num_edges
    clique_red = [0] * len(cliques)
    clique_blue = [0] * len(cliques)
    red_deg = [0] * n
    blue_deg = [0] * n

    def _assign(ek: int, color: int) -> None:
        assignment[ek] = color
        i, j = edges[ek]
        if color == 1:
            red_deg[i] += 1
            red_deg[j] += 1
            for ci in e2c[ek]:
                clique_red[ci] += 1
        else:
            blue_deg[i] += 1
            blue_deg[j] += 1
            for ci in e2c[ek]:
                clique_blue[ci] += 1

    # --- Phase 1: Paley(17) base (guaranteed K₅-free in first 17 vertices) ---
    base = 17
    paley_red = _paley_red_edges(base)
    for i in range(base):
        for j in range(i + 1, base):
            ek = edge_idx[(i, j)]
            color = 1 if (i, j) in paley_red else 0
            _assign(ek, color)

    # --- Phase 2: Shuffled greedy with hub-balance tiebreak for remaining edges ---
    remaining = [ek for ek in range(num_edges) if assignment[ek] == -1]
    rng.shuffle(remaining)

    target_deg = (n - 1) / 2.0

    for ek in remaining:
        i, j = edges[ek]
        best_color = 0
        best_score = float('inf')
        for color in [0, 1]:
            new_v = 0
            for ci in e2c[ek]:
                if color == 1:
                    rc, bc = clique_red[ci] + 1, clique_blue[ci]
                else:
                    rc, bc = clique_red[ci], clique_blue[ci] + 1
                if rc == clique_size_edges or bc == clique_size_edges:
                    new_v += 1
            # Hub-balance tiebreak: penalize imbalance on both endpoints
            if color == 1:
                imbal = abs(red_deg[i] + 1 - target_deg) + abs(red_deg[j] + 1 - target_deg)
            else:
                imbal = abs(blue_deg[i] + 1 - target_deg) + abs(blue_deg[j] + 1 - target_deg)
            # Violations dominate; imbalance only breaks ties (weight 0.001)
            score = new_v + 0.001 * imbal
            if score < best_score:
                best_score = score
                best_color = color
        _assign(ek, best_color)

    return assignment


def save_result(n: int, s: int, assignment: list[int],
                edges: list[tuple[int, int]], out_path: str) -> None:
    """Save valid coloring as IRDME JSON."""
    items = [{"id": str(i), "type": "vertex", "dimensions": {"index": i}} for i in range(n)]
    rels = []
    for k, (i, j) in enumerate(edges):
        rels.append({
            "from": str(i), "to": str(j),
            "type": "red" if assignment[k] == 1 else "blue",
            "directed": False,
        })
    data = {
        "meta": {
            "name": f"Ramsey K_{n} valid coloring, no mono K_{s}",
            "domain": "mathematics", "subdomain": "ramsey_theory",
            "n": n, "s": s,
            "result": f"R({s},{s}) > {n}",
        },
        "items": items,
        "relations": rels,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimized Ramsey local search")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--iter", type=int, default=5000)
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--tabu", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seed-type", choices=["random", "greedy", "structured"], default="greedy")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    n, s = args.n, args.s
    rng = random.Random(args.seed)

    print(f"\n{'='*70}")
    print(f"  RAMSEY LOCAL SEARCH: K_{n}, avoid monochromatic K_{s}")
    print(f"  Restarts: {args.restarts}  Iterations/restart: {args.iter}")
    print(f"  Tabu tenure: {args.tabu}  Noise rate: {args.noise}")
    print(f"{'='*70}\n")

    # Precompute structure (one-time cost)
    t_build = time.time()
    edges, edge_idx, cliques, e2c = build_clique_structure(n, s)
    print(f"  Structure built in {time.time()-t_build:.1f}s\n")

    overall_best = float('inf')
    overall_best_assignment = None
    t_start = time.time()

    for restart in range(args.restarts):
        print(f"\n-- Restart {restart+1}/{args.restarts} "
              f"(elapsed {time.time()-t_start:.0f}s) --")

        # Generate seed
        if args.seed_type == "greedy":
            init = greedy_seed(n, s, rng, edges, cliques, e2c)
            init_v = len([ci for ci, ce in enumerate(cliques)
                          if (sum(init[ek] for ek in ce) == s*(s-1)//2 or
                              sum(1-init[ek] for ek in ce) == s*(s-1)//2)])
            print(f"  Greedy seed: {init_v} initial violations")
        elif args.seed_type == "structured":
            init = structured_seed(n, s, rng, edges, edge_idx, cliques, e2c)
            init_v = len([ci for ci, ce in enumerate(cliques)
                          if (sum(init[ek] for ek in ce) == s*(s-1)//2 or
                              sum(1-init[ek] for ek in ce) == s*(s-1)//2)])
            print(f"  Structured (Paley+FPL) seed: {init_v} initial violations")
        else:
            init = random_seed(n, s, rng)
            print(f"  Random seed")

        result, iters, best_v = local_search(
            n=n, s=s,
            edges=edges, cliques=cliques, e2c=e2c,
            initial_assignment=init,
            max_iter=args.iter,
            tabu_tenure=args.tabu,
            noise_rate=args.noise,
            rng=rng,
            verbose=True,
            report_every=max(1, args.iter // 10),
        )

        if best_v < overall_best:
            overall_best = best_v
            if result is not None:
                overall_best_assignment = result
            print(f"  *** New best: {overall_best} violations ***")

        if result is not None:
            print(f"\n  {'='*60}")
            print(f"  FOUND VALID K_{n} COLORING! R({s},{s}) > {n}")
            print(f"  Total time: {time.time()-t_start:.1f}s")
            print(f"  {'='*60}\n")
            out_path = args.out or f"datasets/RAMSEY_K{n}_S{s}_solution.json"
            save_result(n, s, result, edges, out_path)
            return

    print(f"\n  Search complete. Best violations: {overall_best}")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    if overall_best <= 5:
        print(f"  Close! Try more restarts or longer iterations.")


if __name__ == "__main__":
    main()
