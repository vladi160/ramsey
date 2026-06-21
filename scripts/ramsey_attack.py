"""
Ramsey Number Attack — R(s,s) lower bound search.

Attempts to find a 2-coloring of K_n that avoids monochromatic K_s in both
color classes. If found for n, proves R(s,s) > n.

Current state of art:
  R(4,4) = 18  (exact)
  R(5,5) in [43, 48]  (gap is the open problem)

Strategy:
  1. Start from a structured seed (Paley graph or best known coloring)
  2. Check for violations (monochromatic K_s cliques)
  3. Run WalkSAT-style local search:
       - pick a random violated clique
       - flip the edge with the lowest "break count" (fewest other violations it fixes)
       - FPL tiebreak: prefer flipping edges connecting low-degree (peripheral) nodes
  4. Report: valid coloring found, or best violation count after max iterations

Usage:
  # Verify Paley(17) avoids K_4 in both colors (known: proves R(4,4) >= 18)
  python scripts/ramsey_attack.py --n 17 --s 4 --seed paley --verify

  # Try to find valid K_42 coloring avoiding K_5 (proves R(5,5) >= 43, already known)
  python scripts/ramsey_attack.py --n 42 --s 5 --seed paley --iter 100000

  # The real attack: find valid K_43 coloring (would prove R(5,5) >= 44, NEW RESULT)
  python scripts/ramsey_attack.py --n 43 --s 5 --seed paley --iter 500000

  # Try multiple random seeds and keep best
  python scripts/ramsey_attack.py --n 43 --s 5 --seed random --restarts 20 --iter 200000
"""

from __future__ import annotations
import argparse
import math
import time
import sys
import json
from pathlib import Path

import numpy as np


# ── Paley graph ────────────────────────────────────────────────────────────────

def paley_adjacency(p: int) -> np.ndarray:
    """
    Paley(p) adjacency matrix for prime p ≡ 1 (mod 4).
    Nodes 0..p-1. Edge (i,j) iff (j-i) is a nonzero quadratic residue mod p.
    Self-complementary: the complement is isomorphic to itself.
    """
    assert p % 4 == 1, f"Paley graph requires p ≡ 1 (mod 4), got p={p}"
    # Quadratic residues mod p (excluding 0)
    qr = set((i * i) % p for i in range(1, p))
    adj = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in qr or (i - j) % p in qr:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def circulant_adjacency(n: int, generators: list[int]) -> np.ndarray:
    """
    Circulant graph C(n, S). Edge (i,j) iff |i-j| mod n in S.
    Useful for constructing Ramsey lower bound graphs.
    """
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for g in generators:
            j = (i + g) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return adj


# ── Clique detection ───────────────────────────────────────────────────────────

def find_monochromatic_clique(adj: np.ndarray, size: int) -> list[int] | None:
    """
    Find first monochromatic clique of given size in the red graph (adj=1)
    or return None. Uses branch-and-bound — much faster than C(n,size) brute force.
    """
    n = adj.shape[0]

    def extend(clique: list[int], candidates: list[int]) -> list[int] | None:
        if len(clique) == size:
            return clique
        needed = size - len(clique)
        for i in range(len(candidates) - needed + 1):
            v = candidates[i]
            # Candidates for next level = candidates after v that are also adj to v
            new_cands = [c for c in candidates[i + 1:] if adj[v, c]]
            if 1 + len(new_cands) >= needed:
                result = extend(clique + [v], new_cands)
                if result is not None:
                    return result
        return None

    for start in range(n):
        cands = [j for j in range(start + 1, n) if adj[start, j]]
        result = extend([start], cands)
        if result is not None:
            return result
    return None


def count_violations(adj_red: np.ndarray, size_r: int, size_b: int) -> tuple[int, int]:
    """Count all monochromatic red K_{size_r} and blue K_{size_b} cliques."""
    n = adj_red.shape[0]
    adj_blue = np.ones((n, n), dtype=np.int8) - adj_red
    np.fill_diagonal(adj_blue, 0)
    red_count = _count_cliques(adj_red, size_r, n)
    blue_count = _count_cliques(adj_blue, size_b, n)
    return red_count, blue_count


def _count_cliques(adj: np.ndarray, size: int, n: int) -> int:
    count = 0
    def extend(clique, candidates):
        nonlocal count
        if len(clique) == size:
            count += 1
            return
        needed = size - len(clique)
        for i in range(len(candidates) - needed + 1):
            v = candidates[i]
            new_cands = [c for c in candidates[i + 1:] if adj[v, c]]
            if 1 + len(new_cands) >= needed:
                extend(clique + [v], new_cands)
    for start in range(n):
        cands = [j for j in range(start + 1, n) if adj[start, j]]
        extend([start], cands)
    return count


def find_all_violated_cliques(adj_red: np.ndarray, size: int) -> list[tuple[str, tuple[int, ...]]]:
    """
    Find all monochromatic cliques of given size in red or blue graphs.
    Returns list of ("red"|"blue", tuple_of_nodes).
    """
    n = adj_red.shape[0]
    adj_blue = np.ones((n, n), dtype=np.int8) - adj_red
    np.fill_diagonal(adj_blue, 0)
    violations = []

    def collect(adj, color, clique, candidates):
        if len(clique) == size:
            violations.append((color, tuple(clique)))
            return
        needed = size - len(clique)
        for i in range(len(candidates) - needed + 1):
            v = candidates[i]
            new_cands = [c for c in candidates[i + 1:] if adj[v, c]]
            if 1 + len(new_cands) >= needed:
                collect(adj, color, clique + [v], new_cands)

    for start in range(n):
        r_cands = [j for j in range(start + 1, n) if adj_red[start, j]]
        collect(adj_red, "red", [start], r_cands)
        b_cands = [j for j in range(start + 1, n) if adj_blue[start, j]]
        collect(adj_blue, "blue", [start], b_cands)

    return violations


# ── FPL hub scores ─────────────────────────────────────────────────────────────

def compute_hub_scores(adj_red: np.ndarray) -> np.ndarray:
    """
    Hub score per node = normalized degree centrality in the red graph.
    Used to identify structurally central nodes.
    """
    deg = adj_red.sum(axis=1).astype(float)
    total = deg.sum()
    return deg / (total + 1e-8)


def edge_break_count(adj_red: np.ndarray, violations: list, i: int, j: int, size: int) -> int:
    """
    How many violations would flipping edge (i,j) fix?
    A violation is fixed if (i,j) is in it — flipping changes the monochromatic count.
    """
    return sum(1 for _, clique in violations if i in clique and j in clique)


# ── Local search ───────────────────────────────────────────────────────────────

def steepest_descent(
    adj_red: np.ndarray,
    size: int,
    max_iter: int = 10_000,
    noise_rate: float = 0.05,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
    report_every: int = 500,
) -> tuple[np.ndarray | None, int, int]:
    """
    Steepest descent: at each step, try EVERY possible edge flip and pick
    the one that most reduces total violations. Occasionally add noise moves.

    For K_17 (136 edges, ~30 violations): ~136 × violation-check per step.
    Much more powerful than WalkSAT which only sees violated cliques.

    FPL guidance: when multiple flips tie on violation reduction, prefer
    flipping the edge with the lowest hub score (most peripheral edge).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = adj_red.shape[0]
    adj = adj_red.copy()
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    best_adj = adj.copy()
    best_v = len(find_all_violated_cliques(adj, size))

    hub_scores = compute_hub_scores(adj)

    for it in range(max_iter):
        v_current = len(find_all_violated_cliques(adj, size))

        if v_current == 0:
            if verbose:
                print(f"  SOLVED at iteration {it}!")
            return adj, it, 0

        if v_current < best_v:
            best_v = v_current
            best_adj = adj.copy()
            hub_scores = compute_hub_scores(adj)

        if verbose and it % report_every == 0:
            print(f"  iter {it:>5}  violations={v_current:>3}  best={best_v:>3}")

        if rng.random() < noise_rate:
            # Noise: flip a random edge
            ei, ej = all_edges[rng.integers(len(all_edges))]
            adj[ei, ej] ^= 1; adj[ej, ei] ^= 1
            continue

        # Steepest descent: try all edge flips, pick the best
        best_flip = None
        best_delta = float("inf")  # lower is better
        best_hp = float("inf")

        for ei, ej in all_edges:
            adj[ei, ej] ^= 1; adj[ej, ei] ^= 1
            v_new = len(find_all_violated_cliques(adj, size))
            delta = v_new - v_current
            hp = hub_scores[ei] + hub_scores[ej]

            if delta < best_delta or (delta == best_delta and hp < best_hp):
                best_delta = delta
                best_hp = hp
                best_flip = (ei, ej)

            adj[ei, ej] ^= 1; adj[ej, ei] ^= 1  # undo

        if best_flip is not None:
            ei, ej = best_flip
            adj[ei, ej] ^= 1; adj[ej, ei] ^= 1

    vf = len(find_all_violated_cliques(adj, size))
    return None, max_iter, vf


def tabu_search(
    adj_red: np.ndarray,
    size: int,
    max_iter: int = 500_000,
    tabu_tenure: int = 7,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
    report_every: int = 10_000,
) -> tuple[np.ndarray | None, int, int]:
    """
    Tabu search for Ramsey colorings — the most effective known local search
    for this problem class.

    Key ideas:
    - Short-term memory (tabu list): recently flipped edges cannot be flipped
      again for `tabu_tenure` iterations. Prevents cycling.
    - Aspiration criterion: a tabu edge CAN be flipped if it achieves a new
      global best (fewer violations than ever seen).
    - FPL hub guidance: among edges tied on break count, prefer flipping
      lower-importance (peripheral) edges. Hub-hub edges are the last resort.
    - Restart from best: if stuck for 2*tabu_tenure without improvement,
      restart from the best coloring found so far (not last position).

    The FPL principle justifies tabu search structure: hubs are stable across
    perturbations; peripheral edges absorb local adjustments.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = adj_red.shape[0]
    adj = adj_red.copy()
    adj_blue = (np.ones((n, n), dtype=np.int8) - adj)
    np.fill_diagonal(adj_blue, 0)

    best_adj = adj.copy()
    best_violations = float("inf")

    tabu_list: dict[tuple[int, int], int] = {}  # edge -> iteration when tabu expires
    stuck_since = 0

    hub_scores = compute_hub_scores(adj)

    for it in range(max_iter):
        violations = find_all_violated_cliques(adj, size)
        v_count = len(violations)

        if v_count < best_violations:
            best_violations = v_count
            best_adj = adj.copy()
            hub_scores = compute_hub_scores(adj)
            stuck_since = it

        if not violations:
            if verbose:
                print(f"  SOLVED at iteration {it} — valid coloring found!")
            return adj, it, 0

        if verbose and it % report_every == 0:
            print(f"  iter {it:>7}  violations={v_count:>4}  best={int(best_violations):>4}  "
                  f"tabu_size={len(tabu_list)}")

        # Iterated Local Search: when stuck, perturb best solution and restart
        stuck_thresh = max(2000, tabu_tenure * 30)
        if it - stuck_since > stuck_thresh:
            # Perturbation: flip several edges of the best coloring to escape basin
            adj = best_adj.copy()
            perturb_count = max(3, int(n * 0.07))  # perturb ~7% of vertices' edges
            nodes_to_perturb = rng.choice(n, size=perturb_count, replace=False)
            for p_node in nodes_to_perturb:
                j = rng.integers(n)
                if j != p_node:
                    adj[p_node, j] ^= 1; adj[j, p_node] ^= 1
            adj_blue = (np.ones((n, n), dtype=np.int8) - adj)
            np.fill_diagonal(adj_blue, 0)
            tabu_list.clear()
            stuck_since = it
            hub_scores = compute_hub_scores(adj)
            if verbose and it % report_every < stuck_thresh:
                print(f"  [ILS perturb at iter {it}: flipped {perturb_count} edges from best]")
            continue

        # Pick a random violated clique
        _, clique = violations[rng.integers(v_count)]
        edges_in_clique = [
            (clique[a], clique[b])
            for a in range(len(clique))
            for b in range(a + 1, len(clique))
        ]

        # Score each edge: (break_count, hub_penalty, is_tabu)
        best_score = None
        chosen = None
        for ei, ej in edges_in_clique:
            edge_key = (min(ei, ej), max(ei, ej))
            bc = sum(1 for _, cl in violations if ei in cl and ej in cl)
            hp = hub_scores[ei] + hub_scores[ej]  # lower = more peripheral
            is_tabu = tabu_list.get(edge_key, 0) > it

            # Aspiration: allow tabu if it would set a new best
            allows_aspiration = is_tabu and (v_count - bc) < best_violations

            if is_tabu and not allows_aspiration:
                continue

            # Score tuple: maximize break_count, minimize hub_penalty
            score = (bc, -hp)
            if best_score is None or score > best_score:
                best_score = score
                chosen = edge_key

        if chosen is None:
            # All edges tabu — pick random (emergency escape)
            ei, ej = edges_in_clique[rng.integers(len(edges_in_clique))]
            chosen = (min(ei, ej), max(ei, ej))

        ei, ej = chosen
        adj[ei, ej] ^= 1; adj[ej, ei] ^= 1
        adj_blue[ei, ej] ^= 1; adj_blue[ej, ei] ^= 1

        # Mark edge as tabu
        tabu_list[chosen] = it + tabu_tenure

        # Clean up expired tabu entries periodically
        if it % 1000 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > it}

    vf = len(find_all_violated_cliques(adj, size))
    if verbose:
        print(f"  Search ended. Best violations={int(best_violations)}  Final={vf}")
    return None, max_iter, int(best_violations)


def walksaT_search(
    adj_red: np.ndarray,
    size: int,
    max_iter: int = 100_000,
    noise: float = 0.1,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
    report_every: int = 5000,
) -> tuple[np.ndarray | None, int, int]:
    """
    Search to eliminate monochromatic K_size cliques.

    Two modes controlled by `temperature`:
      temperature=0: WalkSAT (greedy + noise).
      temperature>0: Simulated annealing — accepts worsening moves with
                     probability exp(-delta/T), T decays over time.

    FPL guidance: when multiple edges have the same break count, prefer
    flipping edges connecting lower-degree (peripheral) nodes. Hub-hub
    edges are preserved last — they carry the structural skeleton that
    makes valid colorings stable.

    Returns: (adj_if_solved, iterations, final_violation_count)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    adj = adj_red.copy()
    n = adj.shape[0]
    best_violations = float("inf")
    best_adj = adj.copy()

    T0 = temperature
    hub_scores = compute_hub_scores(adj)

    for it in range(max_iter):
        violations = find_all_violated_cliques(adj, size)

        if not violations:
            if verbose:
                print(f"  SOLVED at iteration {it} — valid coloring found!")
            return adj, it, 0

        v_count = len(violations)
        if v_count < best_violations:
            best_violations = v_count
            best_adj = adj.copy()
            hub_scores = compute_hub_scores(adj)

        if verbose and it % report_every == 0:
            T_now = T0 * (1 - it / max_iter) if T0 > 0 else 0
            print(f"  iter {it:>7}  violations={v_count:>4}  best={int(best_violations):>4}  "
                  f"T={T_now:.3f}  max_hub={hub_scores.max():.4f}")

        # Pick a random violated clique
        _, clique = violations[rng.integers(len(violations))]
        edges = [(clique[a], clique[b]) for a in range(len(clique))
                 for b in range(a + 1, len(clique))]

        if T0 > 0:
            # Simulated annealing
            T = T0 * (1 - it / max_iter)
            ei, ej = edges[rng.integers(len(edges))]
            # Accept based on delta violations
            adj[ei, ej] ^= 1; adj[ej, ei] ^= 1
            new_violations = find_all_violated_cliques(adj, size)
            delta = len(new_violations) - v_count
            if delta > 0 and T > 1e-9 and rng.random() >= math.exp(-delta / T):
                # Reject: undo
                adj[ei, ej] ^= 1; adj[ej, ei] ^= 1
        elif rng.random() < noise:
            # WalkSAT noise: random edge in violated clique
            ei, ej = edges[rng.integers(len(edges))]
            adj[ei, ej] ^= 1; adj[ej, ei] ^= 1
        else:
            # WalkSAT greedy: pick edge that fixes the most violations
            best_break = -1
            best_hub_penalty = float("inf")
            chosen = edges[0]
            for ei, ej in edges:
                bc = sum(1 for _, cl in violations if ei in cl and ej in cl)
                hp = hub_scores[ei] + hub_scores[ej]
                if bc > best_break or (bc == best_break and hp < best_hub_penalty):
                    best_break = bc
                    best_hub_penalty = hp
                    chosen = (ei, ej)
            ei, ej = chosen
            adj[ei, ej] ^= 1; adj[ej, ei] ^= 1

    if verbose:
        vf = len(find_all_violated_cliques(adj, size))
        print(f"  Search ended. Best violations={int(best_violations)}  Final={vf}")

    return None, max_iter, int(best_violations)


# ── Seeding strategies ─────────────────────────────────────────────────────────

def seed_paley(n: int) -> np.ndarray:
    """
    Use Paley(p) as seed for n ≤ p. Picks the smallest prime p ≡ 1 (mod 4) ≥ n.
    If n > p: embed the Paley graph and assign remaining edges randomly.
    """
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0: return False
        return True

    p = n if (is_prime(n) and n % 4 == 1) else None
    if p is None:
        # Find nearest prime ≡ 1 (mod 4) ≤ n
        for q in range(n, 1, -1):
            if is_prime(q) and q % 4 == 1:
                p = q
                break

    if p is None or p < 5:
        return seed_random(n, seed=0)

    adj_p = paley_adjacency(p)

    if n == p:
        return adj_p

    # Extend: embed Paley(p) in top-left, assign new edges balanced
    adj = np.zeros((n, n), dtype=np.int8)
    adj[:p, :p] = adj_p

    # New vertices get alternating connections to balance red/blue degree
    rng = np.random.default_rng(42)
    for i in range(p, n):
        for j in range(i):
            # Balance: if j already has more red neighbors, give blue connection
            red_deg_j = adj[:i, j].sum() + adj[j, :i].sum()
            if j < p:
                target_red = p // 2  # roughly half
            else:
                target_red = i // 2
            if red_deg_j < target_red:
                adj[i, j] = 1
                adj[j, i] = 1
            elif red_deg_j == target_red and rng.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1

    return adj


def seed_extend(n: int, size: int, prev_path: str, seed: int = 42) -> np.ndarray | None:
    """
    Seed K_n coloring by extending a valid K_{n-1} coloring loaded from file.
    Add the new vertex n-1 by greedily assigning its edges to existing nodes.
    This is the strongest seed: a valid coloring of K_{n-1} gives only n-1
    new edges to color, starting from a near-valid state.
    """
    try:
        with open(prev_path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [extend seed] Cannot load {prev_path}: {e}")
        return None

    m = n - 1  # size of the previously solved graph
    adj = np.zeros((n, n), dtype=np.int8)
    adj_blue = np.zeros((n, n), dtype=np.int8)

    # Load existing coloring into top-left m×m block
    for rel in data.get("relations", []):
        i, j = int(rel["from"]), int(rel["to"])
        if i < m and j < m:
            if rel.get("type") == "red":
                adj[i, j] = 1; adj[j, i] = 1
            else:
                adj_blue[i, j] = 1; adj_blue[j, i] = 1

    rng = np.random.default_rng(seed)
    new_v = m  # the new vertex index

    # Greedy assignment for new vertex's n-1 edges
    for existing in range(m):
        # Count new violations if we add (new_v, existing) as red
        adj[new_v, existing] = 1; adj[existing, new_v] = 1
        common_red = [k for k in range(m) if k != existing and adj[new_v, k] and adj[existing, k]]
        # Check for (size-1) cliques in red involving both new_v and existing
        adj[new_v, existing] = 0; adj[existing, new_v] = 0
        adj_tmp = adj.copy()
        adj_tmp[new_v, existing] = 1; adj_tmp[existing, new_v] = 1
        red_cost = sum(1 for k in range(m) if k != existing
                      and adj_tmp[new_v, k] and adj_tmp[existing, k])

        # Count new violations if we add as blue
        adj_blue[new_v, existing] = 1; adj_blue[existing, new_v] = 1
        blue_cost = sum(1 for k in range(m) if k != existing
                       and adj_blue[new_v, k] and adj_blue[existing, k])
        adj_blue[new_v, existing] = 0; adj_blue[existing, new_v] = 0

        # Assign cheaper color
        if red_cost <= blue_cost:
            adj[new_v, existing] = 1; adj[existing, new_v] = 1
        else:
            adj_blue[new_v, existing] = 1; adj_blue[existing, new_v] = 1

    print(f"  [extend seed] Loaded {prev_path} and extended to K_{n}")
    return adj


def seed_random(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def seed_greedy(n: int, size: int, seed: int = 42) -> np.ndarray:
    """
    Build coloring greedily: process edges one at a time, picking the color
    that creates fewest new monochromatic cliques among ALREADY-COLORED edges.

    Critical: only currently assigned edges count — uncolored edges are ignored.
    This gives a valid lower bound on violations at each step.
    """
    rng = np.random.default_rng(seed)
    adj_red = np.zeros((n, n), dtype=np.int8)
    adj_blue = np.zeros((n, n), dtype=np.int8)

    def count_new_cliques_through_uv(adj: np.ndarray, u: int, v: int, clique_size: int) -> int:
        """Count cliques of given size through edge (u,v) in adj (where u,v already connected)."""
        # Find common neighbors of u and v in adj
        nbrs_u = np.where(adj[u] > 0)[0]
        nbrs_v = np.where(adj[v] > 0)[0]
        common = list(set(nbrs_u) & set(nbrs_v) - {u, v})
        needed = clique_size - 2
        if len(common) < needed:
            return 0
        if needed == 1:
            return len(common)
        # Count (needed)-cliques in the subgraph induced by common
        count = 0
        def extend(clique, cands):
            nonlocal count
            if len(clique) == needed:
                count += 1
                return
            remain = needed - len(clique)
            for i2 in range(len(cands) - remain + 1):
                w = cands[i2]
                new_cands = [c for c in cands[i2+1:] if adj[w, c]]
                if 1 + len(new_cands) >= remain:
                    extend(clique + [w], new_cands)
        extend([], common)
        return count

    # Process edges in shuffled order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)

    for u, v in edges:
        # Try red: temporarily add (u,v) to red graph
        adj_red[u, v] = 1; adj_red[v, u] = 1
        red_cost = count_new_cliques_through_uv(adj_red, u, v, size)
        adj_red[u, v] = 0; adj_red[v, u] = 0

        # Try blue: temporarily add (u,v) to blue graph
        adj_blue[u, v] = 1; adj_blue[v, u] = 1
        blue_cost = count_new_cliques_through_uv(adj_blue, u, v, size)
        adj_blue[u, v] = 0; adj_blue[v, u] = 0

        # Assign the cheaper color; random tiebreak
        if red_cost < blue_cost:
            adj_red[u, v] = 1; adj_red[v, u] = 1
        elif blue_cost < red_cost:
            adj_blue[u, v] = 1; adj_blue[v, u] = 1
        else:
            if rng.random() < 0.5:
                adj_red[u, v] = 1; adj_red[v, u] = 1
            else:
                adj_blue[u, v] = 1; adj_blue[v, u] = 1

    return adj_red


# ── Output ─────────────────────────────────────────────────────────────────────

def save_coloring(adj_red: np.ndarray, path: str, n: int, s: int) -> None:
    """Save a valid coloring as IRDME JSON for FPL analysis."""
    items = [{"id": str(i), "type": "vertex", "dimensions": {"value": i}} for i in range(n)]
    relations = []
    for i in range(n):
        for j in range(i + 1, n):
            color = "red" if adj_red[i, j] else "blue"
            relations.append({
                "from": str(i), "to": str(j),
                "type": color, "directed": False,
                "dimensions": {"color_value": 1 if color == "red" else 0}
            })
    data = {
        "meta": {
            "name": f"Ramsey Coloring K_{n} avoiding K_{s}",
            "description": f"Valid 2-coloring of K_{n} with no monochromatic K_{s}. Proves R({s},{s}) > {n}.",
            "domain": "mathematics",
            "subdomain": "ramsey_theory",
            "version": "1.0",
            "source": "scripts/ramsey_attack.py",
            "n": n, "s": s,
        },
        "items": items,
        "relations": relations,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Coloring saved -> {path}")
    print(f"  Verify with: python irdme.py law {path}")
    print(f"  Analyze with: python irdme.py hubs {path}")


def print_adjacency_summary(adj_red: np.ndarray) -> None:
    n = adj_red.shape[0]
    red_deg = adj_red.sum(axis=1)
    adj_blue = np.ones((n, n), dtype=np.int8) - adj_red
    np.fill_diagonal(adj_blue, 0)
    blue_deg = adj_blue.sum(axis=1)
    print(f"  n={n}  edges={adj_red.sum()//2} red + {adj_blue.sum()//2} blue = {n*(n-1)//2} total")
    print(f"  red degree:  min={red_deg.min()} max={red_deg.max()} mean={red_deg.mean():.1f}")
    print(f"  blue degree: min={blue_deg.min()} max={blue_deg.max()} mean={blue_deg.mean():.1f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ramsey number attack: find 2-colorings of K_n avoiding monochromatic K_s.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n", type=int, required=True, help="Graph size (K_n)")
    parser.add_argument("--s", type=int, default=5, help="Clique size to avoid (default: 5 for R(5,5))")
    parser.add_argument("--seed", choices=["paley", "random", "greedy", "extend"], default="greedy",
                        help="Seed strategy (default: greedy). 'extend' loads --from-file and adds one vertex.")
    parser.add_argument("--from-file", default=None, dest="from_file",
                        help="For --seed extend: path to previous solution JSON (auto-detected if omitted)")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify the seed coloring, no local search")
    parser.add_argument("--iter", type=int, default=100_000, help="Max search iterations (default: 100000)")
    parser.add_argument("--mode", choices=["tabu", "walksaT", "annealing", "steepest"], default="tabu",
                        help="Search mode (default: tabu)")
    parser.add_argument("--noise", type=float, default=0.1, help="WalkSAT noise (0.0=greedy, default: 0.1)")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Simulated annealing start temperature (default: 2.0)")
    parser.add_argument("--tabu-tenure", type=int, default=7, dest="tabu_tenure",
                        help="Tabu tenure — how long an edge stays forbidden (default: 7)")
    parser.add_argument("--restarts", type=int, default=1, help="Number of random restarts (default: 1)")
    parser.add_argument("--out", default=None, help="Save valid coloring to JSON file")
    parser.add_argument("--report-every", type=int, default=5000, dest="report_every")
    args = parser.parse_args()

    n, s = args.n, args.s
    print()
    print("=" * 70)
    print(f"  RAMSEY ATTACK: K_{n}, avoiding monochromatic K_{s}")
    print(f"  Goal: prove R({s},{s}) > {n}" + (" — NEW RESULT" if n >= 43 else ""))
    print("=" * 70)
    print()

    global_start = time.time()
    solved_adj = None

    for restart in range(args.restarts):
        if args.restarts > 1:
            print(f"-- Restart {restart+1}/{args.restarts} --")

        # Build seed
        seed_t = time.time()
        if args.seed == "paley":
            adj = seed_paley(n)
            print(f"  Seed: Paley-based ({time.time()-seed_t:.2f}s)")
        elif args.seed == "greedy":
            adj = seed_greedy(n, s, seed=restart * 137 + 42)
            print(f"  Seed: greedy construction ({time.time()-seed_t:.2f}s)")
        elif args.seed == "extend":
            prev = args.from_file or f"datasets/RAMSEY_K{n-1}_S{s}.json"
            adj_ext = seed_extend(n, s, prev, seed=restart * 137 + 42)
            if adj_ext is None:
                print(f"  Extend failed — falling back to greedy")
                adj = seed_greedy(n, s, seed=restart * 137 + 42)
            else:
                adj = adj_ext
            print(f"  Seed: extend from K_{n-1} ({time.time()-seed_t:.2f}s)")
        else:
            adj = seed_random(n, seed=restart * 137 + 42)
            print(f"  Seed: random (seed={restart * 137 + 42})")

        print_adjacency_summary(adj)
        print()

        # Check seed violations
        check_t = time.time()
        violations = find_all_violated_cliques(adj, s)
        print(f"  Seed violations (mono K_{s}): {len(violations)}  ({time.time()-check_t:.2f}s)")

        if args.verify:
            if not violations:
                print(f"\n  VALID: Seed coloring of K_{n} has no monochromatic K_{s}.")
                print(f"  This proves R({s},{s}) > {n}.")
                if args.out:
                    save_coloring(adj, args.out, n, s)
            else:
                red_v = sum(1 for c, _ in violations if c == "red")
                blue_v = sum(1 for c, _ in violations if c == "blue")
                print(f"\n  NOT VALID: {len(violations)} violations ({red_v} red, {blue_v} blue).")
                print(f"  Red K_{s} example: {next((cl for c, cl in violations if c=='red'), 'none')}")
                print(f"  Blue K_{s} example: {next((cl for c, cl in violations if c=='blue'), 'none')}")
            return

        if not violations:
            print(f"\n  Seed is already valid! No local search needed.")
            solved_adj = adj
            break

        # Local search
        mode_desc = {
            "tabu": f"Tabu search (tenure={args.tabu_tenure}, FPL hub guidance)",
            "walksaT": f"WalkSAT (noise={args.noise})",
            "annealing": f"Simulated annealing (T0={args.temperature})",
            "steepest": f"Steepest descent (noise={args.noise}, FPL hub tiebreak)",
        }[args.mode]
        print(f"\n  Starting {mode_desc}, max_iter={args.iter:,}")
        print(f"  FPL: peripheral edges flipped first; hub-hub edges preserved last.")
        print()
        if args.mode == "steepest":
            result, iters, final_v = steepest_descent(
                adj, s,
                max_iter=args.iter,
                noise_rate=args.noise,
                rng=np.random.default_rng(restart * 1000 + 42),
                verbose=True,
                report_every=args.report_every,
            )
        elif args.mode == "tabu":
            result, iters, final_v = tabu_search(
                adj, s,
                max_iter=args.iter,
                tabu_tenure=args.tabu_tenure,
                rng=np.random.default_rng(restart * 1000 + 42),
                verbose=True,
                report_every=args.report_every,
            )
        else:
            result, iters, final_v = walksaT_search(
                adj, s,
                max_iter=args.iter,
                noise=args.noise,
                temperature=args.temperature if args.mode == "annealing" else 0.0,
                rng=np.random.default_rng(restart * 1000 + 42),
                verbose=True,
                report_every=args.report_every,
            )
        if result is not None:
            solved_adj = result
            break
        print()

    elapsed = time.time() - global_start
    print()
    print("=" * 70)
    if solved_adj is not None:
        print(f"  RESULT: VALID coloring found for K_{n} (no monochromatic K_{s})")
        print(f"  MATHEMATICAL CLAIM: R({s},{s}) > {n}")
        print_adjacency_summary(solved_adj)
        print(f"  Time: {elapsed:.1f}s")
        if args.out:
            save_coloring(solved_adj, args.out, n, s)
        else:
            default_out = f"datasets/RAMSEY_K{n}_S{s}.json"
            save_coloring(solved_adj, default_out, n, s)
    else:
        print(f"  RESULT: No valid coloring found for K_{n} in {elapsed:.1f}s")
        print(f"  Best achieved: {final_v} violations remaining")
        print(f"  This does NOT prove R({s},{s}) <= {n}.")
        print(f"  Try: --iter {args.iter * 2}  or  --restarts {args.restarts + 4}  or  --noise 0.15")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
