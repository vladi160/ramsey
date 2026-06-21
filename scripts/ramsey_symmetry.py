"""
ramsey_symmetry.py -- Symmetry-restricted Ramsey search for K_42

The navigator experiment showed that pure signature navigation doesn't reduce
violations. This script tries a different structural restriction:

Z_21-SYMMETRIC (BI-CIRCULANT) SEARCH
======================================
K_42 vertices: Z_21 × {0,1}  (two copies of Z_21)
  vertex i  →  (pos=i%21, half=i//21)

Z_21 action: rotate positions in BOTH halves simultaneously
  (pos, half) → ((pos+1) % 21, half)

This partitions all 861 K_42 edges into exactly 41 orbit types (21 edges each):
  A_k  (k=1..10): within half-0, position difference k  [type 0-9]
  B_k  (k=1..10): within half-1, position difference k  [type 10-19]
  C_k  (k=0..20): between halves, relative diff k       [type 20-40]

A bi-circulant 2-coloring assigns each type a single color (0/1).
41 free bits instead of 861.  Search space: 2^41 ~= 2T.

SELF-COMPLEMENTARY MODE (sc):
  Forces B_k = 1 - A_k for k=1..10. Only 31 free bits: A_1..A_10 + C_0..C_20.
  Each "A-flip" move flips A_k and B_k simultaneously.
  The 84v bicirculant solution is 8/10 self-complementary; SC subspace may
  contain a deeper minimum.

Usage:
    # Bi-circulant tabu search (best result: 84v)
    python scripts/ramsey_symmetry.py --mode bicirculant --steps 10000
    python scripts/ramsey_symmetry.py --mode bicirculant --steps 100000 --tabu 8

    # Self-complementary search (31 free bits instead of 41)
    python scripts/ramsey_symmetry.py --mode sc --steps 100000
    python scripts/ramsey_symmetry.py --mode sc --steps 100000 --from-checkpoint datasets/RAMSEY_K42_S5_bicirculant_84v.json

    # Paley warm-start
    python scripts/ramsey_symmetry.py --mode paley
"""

import json, math, random, time, argparse, itertools, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Force line-buffered stdout so background-task output files update in real time
sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Orbit utilities
# ---------------------------------------------------------------------------

N   = 42
N1  = 21       # half size
N_TYPES = 41   # 10 + 10 + 21

def vertex_decompose(v):
    return v % N1, v // N1   # (pos, half)

def edge_orbit(u, v):
    """Return orbit type index for edge (u,v), u < v. Range [0, 40]."""
    pu, hu = vertex_decompose(u)
    pv, hv = vertex_decompose(v)
    if hu == hv:
        diff = (pv - pu) % N1
        k = min(diff, N1 - diff)          # k in 1..10
        return hu * 10 + (k - 1)          # 0-9 or 10-19
    else:
        # Normalize: half-0 is 'u', half-1 is 'v'
        if hu == 1:
            pu, pv = pv, pu
        diff = (pv - pu) % N1             # 0..20
        return 20 + diff                   # 20-40


def orbit_edges(t):
    """Return all 21 edges in orbit type t."""
    edges = []
    if t < 10:
        # Within half-0, difference k = t+1
        k = t + 1
        for pos in range(N1):
            u = pos
            v = (pos + k) % N1
            if u > v: u, v = v, u
            edges.append((u, v))
    elif t < 20:
        # Within half-1, difference k = t-9
        k = t - 9
        for pos in range(N1):
            u = N1 + pos
            v = N1 + (pos + k) % N1
            if u > v: u, v = v, u
            edges.append((u, v))
    else:
        # Between halves, C_{k} where k = t-20
        k = t - 20
        for pos in range(N1):
            u = pos               # half-0
            v = N1 + (pos + k) % N1  # half-1
            edges.append((u, v))
    return edges


def assignment_from_types(types):
    """Convert 41-type coloring to full 861-edge assignment (Rust format)."""
    assignment = [0] * (N * (N-1) // 2)
    idx = 0
    for a in range(N):
        for b in range(a+1, N):
            t = edge_orbit(a, b)
            assignment[idx] = int(types[t])
            idx += 1
    return assignment


# ---------------------------------------------------------------------------
# Precomputation: 5-tuple → edge orbits, type → affected 5-tuples
# ---------------------------------------------------------------------------

def build_lookup_tables(verbose=True):
    """
    Enumerate all C(42,5)=850,668 five-tuples. For each tuple, determine
    its canonical orbit key (minimum rotation) and its 10 edge orbit types.

    Optimization: instead of trying all 21 rotations, only try shifts that
    bring one of the 5 vertices to position 0. At most 5 trials per tuple.
    The canonical key always has 0 as its minimum vertex index (half=0,pos=0)
    for tuples containing a half-0 vertex, or 21 (half=1,pos=0) for half-1-only.

    Returns:
      canon_list: list of [multiplicity, [10 orbit type indices]] per orbit
      type_to_canons: type_to_canons[t] = list of orbit indices containing type t
    """
    if verbose:
        print("  Precomputing 5-tuple lookup tables (5-rotation method)... ",
              end="", flush=True)
    t0 = time.time()

    # Precompute rotation table: rot_vtx[v][k] = vertex index after shift k
    rot_vtx = [[(v % N1 + k) % N1 + (v // N1) * N1 for k in range(N1)]
               for v in range(N)]

    canon_map = {}   # orbit_key -> [multiplicity, edge_types]

    for combo in itertools.combinations(range(N), 5):
        # Try shifts that bring each vertex's position to 0; pick min result
        best = None
        for v in combo:
            shift = (N1 - v % N1) % N1   # shift that brings v's pos to 0
            candidate = tuple(sorted(rot_vtx[w][shift] for w in combo))
            if best is None or candidate < best:
                best = candidate

        if best in canon_map:
            canon_map[best][0] += 1
        else:
            etypes = [edge_orbit(best[i], best[j])
                      for i in range(5) for j in range(i+1, 5)]
            canon_map[best] = [1, etypes]

    canon_list = list(canon_map.values())

    # Precompute numpy arrays for fast bitwise violation counting.
    # mask[idx] = OR of (1 << t) for each edge type t in canonical 5-tuple idx.
    # A 5-tuple is monochromatic iff (assignment_int & mask) in {0, mask}.
    n_orbits = len(canon_list)
    masks_np = np.zeros(n_orbits, dtype=np.int64)
    mults_np = np.zeros(n_orbits, dtype=np.int64)
    for idx, (mult, etypes) in enumerate(canon_list):
        mults_np[idx] = mult
        m = 0
        for t in etypes:
            m |= (1 << t)
        masks_np[idx] = m

    type_to_canons_np = []
    type_to_canons    = [[] for _ in range(N_TYPES)]
    for idx, (mult, etypes) in enumerate(canon_list):
        for t in set(etypes):
            type_to_canons[t].append(idx)
    for t in range(N_TYPES):
        type_to_canons_np.append(np.array(type_to_canons[t], dtype=np.int32))

    # Precompute union arrays for SC mode: A-flip k touches types k AND k+10
    sc_a_canons_np = [
        np.union1d(type_to_canons_np[k], type_to_canons_np[k + 10])
        for k in range(10)
    ]

    elapsed = time.time() - t0
    if verbose:
        print(f"done in {elapsed:.1f}s  ({n_orbits} canonical 5-tuples)")
    return canon_list, masks_np, mults_np, type_to_canons_np, sc_a_canons_np


# ---------------------------------------------------------------------------
# Violation counting (numpy bitwise — fast)
# ---------------------------------------------------------------------------

def types_to_int(types):
    """Pack 41-bit types list into a Python int."""
    v = 0
    for t, c in enumerate(types):
        if c:
            v |= (1 << t)
    return v

def count_violations_np(assignment_int, masks_np, mults_np):
    """Count K5 violations using numpy vectorized bitwise ops. O(40508) array ops."""
    masked = masks_np & assignment_int
    is_mono = (masked == 0) | (masked == masks_np)
    return int(np.sum(mults_np[is_mono]))

def delta_violations_np(assignment_int, flip_t, masks_np, mults_np, type_to_canons_np):
    """
    Compute change in violation count if we flip orbit type flip_t.
    Vectorized over the ~9k canonical 5-tuples containing that type.
    """
    idxs        = type_to_canons_np[flip_t]
    sub_masks   = masks_np[idxs]
    sub_mults   = mults_np[idxs]
    new_int     = assignment_int ^ (1 << flip_t)

    masked_b    = sub_masks & assignment_int
    mono_b      = (masked_b == 0) | (masked_b == sub_masks)

    masked_a    = sub_masks & new_int
    mono_a      = (masked_a == 0) | (masked_a == sub_masks)

    return int(np.sum(sub_mults * (mono_a.astype(np.int64) - mono_b.astype(np.int64))))


def delta_violations_sc(assignment_int, flip_idx, is_a_flip,
                        masks_np, mults_np, type_to_canons_np, sc_a_canons_np):
    """
    Delta for a self-complementary move.
    is_a_flip=True:  flip A_k (type flip_idx) AND B_k (type flip_idx+10) together.
    is_a_flip=False: flip C_j (type flip_idx+20) only.
    sc_a_canons_np: precomputed union arrays for each A-flip (avoids np.union1d per call).
    """
    if is_a_flip:
        k = flip_idx
        idxs = sc_a_canons_np[k]
        new_int = assignment_int ^ (1 << k) ^ (1 << (k + 10))
    else:
        t = flip_idx + 20
        idxs = type_to_canons_np[t]
        new_int = assignment_int ^ (1 << t)

    sub_masks = masks_np[idxs]
    sub_mults = mults_np[idxs]
    masked_b  = sub_masks & assignment_int
    mono_b    = (masked_b == 0) | (masked_b == sub_masks)
    masked_a  = sub_masks & new_int
    mono_a    = (masked_a == 0) | (masked_a == sub_masks)
    return int(np.sum(sub_mults * (mono_a.astype(np.int64) - mono_b.astype(np.int64))))


# ---------------------------------------------------------------------------
# Tabu search on bi-circulant types
# ---------------------------------------------------------------------------

def tabu_search_bicirculant(types, masks_np, mults_np, type_to_canons_np,
                             steps, tabu_tenure, rng, log_every=1000, verbose=True):
    assignment_int = types_to_int(types)
    best_types = list(types)
    best_int   = assignment_int
    best_viols = count_violations_np(assignment_int, masks_np, mults_np)
    cur_viols  = best_viols

    tabu = [-tabu_tenure] * N_TYPES
    step = 0

    if verbose:
        print(f"  {'step':>8}  {'violations':>10}  {'best':>6}  {'steps/s':>8}")
        print("  " + "-"*42)
    t_block = time.time()

    while step < steps:
        # Evaluate all 41 type-flips; pick best non-tabu
        best_delta = None
        best_t = -1
        for t in range(N_TYPES):
            is_tabu = step < tabu[t]
            d = delta_violations_np(assignment_int, t, masks_np, mults_np, type_to_canons_np)
            aspiration = (cur_viols + d < best_viols)
            if is_tabu and not aspiration:
                continue
            if best_delta is None or d < best_delta:
                best_delta = d
                best_t = t

        if best_t == -1:
            best_t = rng.randrange(N_TYPES)
            best_delta = delta_violations_np(assignment_int, best_t, masks_np, mults_np, type_to_canons_np)

        # Apply flip
        assignment_int ^= (1 << best_t)
        types[best_t] ^= 1
        cur_viols += best_delta
        tabu[best_t] = step + tabu_tenure
        step += 1

        if cur_viols < best_viols:
            best_viols = cur_viols
            best_types = list(types)
            best_int   = assignment_int
            if verbose:
                print(f"  ** NEW BEST: {best_viols} violations at step {step}")

        if verbose and step % log_every == 0:
            elapsed_block = time.time() - t_block
            rate = log_every / max(elapsed_block, 0.001)
            t_block = time.time()
            print(f"  {step:>8d}  {cur_viols:>10d}  {best_viols:>6d}  {rate:>7.0f}/s")

    if verbose:
        print(f"\n  Final:  violations={cur_viols}  best={best_viols}")
    return best_types, best_viols


def types_from_assignment(assignment):
    """Derive bi-circulant types from a full 861-edge assignment.
    Assumes the coloring is bi-circulant; takes color of first edge per orbit."""
    types = [None] * N_TYPES
    idx = 0
    for a in range(N):
        for b in range(a+1, N):
            t = edge_orbit(a, b)
            if types[t] is None:
                types[t] = assignment[idx]
            idx += 1
    return [c if c is not None else 0 for c in types]


def tabu_search_sc(types, masks_np, mults_np, type_to_canons_np, sc_a_canons_np,
                   steps, tabu_tenure, rng, log_every=1000, hi_mode=False):
    """
    Symmetry-restricted tabu search with 31 free bits.
    sc_mode (hi_mode=False): B_k = 1 - A_k for k=0..9  (self-complementary)
    hi_mode (hi_mode=True):  B_k = A_k for k=0..9      (half-identical)
    Both use the same move set: 10 A-flips (flip A_k AND B_k) + 21 C-flips.
    """
    # Enforce constraint on initial types
    for k in range(10):
        types[k + 10] = types[k] if hi_mode else (1 - types[k])

    label = "HI" if hi_mode else "SC"
    constraint = "B_k = A_k" if hi_mode else "B_k = 1-A_k"

    assignment_int = types_to_int(types)
    best_types = list(types)
    best_viols = count_violations_np(assignment_int, masks_np, mults_np)
    cur_viols  = best_viols

    tabu_a = [-tabu_tenure] * 10
    tabu_c = [-tabu_tenure] * 21
    step   = 0

    print(f"  {label} start: {best_viols} violations  ({constraint}; 31 free bits: 10A + 21C)")
    print(f"  {'step':>8}  {'violations':>10}  {'best':>6}  {'steps/s':>8}")
    print("  " + "-"*42)
    t_block = time.time()

    while step < steps:
        best_delta = None
        best_move  = None  # (is_a_flip, idx)

        for k in range(10):
            is_tabu = step < tabu_a[k]
            d = delta_violations_sc(assignment_int, k, True,
                                    masks_np, mults_np, type_to_canons_np, sc_a_canons_np)
            aspiration = (cur_viols + d < best_viols)
            if is_tabu and not aspiration:
                continue
            if best_delta is None or d < best_delta:
                best_delta = d
                best_move  = (True, k)

        for j in range(21):
            is_tabu = step < tabu_c[j]
            d = delta_violations_sc(assignment_int, j, False,
                                    masks_np, mults_np, type_to_canons_np, sc_a_canons_np)
            aspiration = (cur_viols + d < best_viols)
            if is_tabu and not aspiration:
                continue
            if best_delta is None or d < best_delta:
                best_delta = d
                best_move  = (False, j)

        if best_move is None:
            if rng.random() < 0.5:
                k = rng.randrange(10)
                best_move  = (True, k)
                best_delta = delta_violations_sc(assignment_int, k, True,
                                                 masks_np, mults_np, type_to_canons_np, sc_a_canons_np)
            else:
                j = rng.randrange(21)
                best_move  = (False, j)
                best_delta = delta_violations_sc(assignment_int, j, False,
                                                 masks_np, mults_np, type_to_canons_np, sc_a_canons_np)

        is_a_flip, idx = best_move
        if is_a_flip:
            assignment_int ^= (1 << idx) | (1 << (idx + 10))
            types[idx]      ^= 1
            types[idx + 10] ^= 1
            tabu_a[idx] = step + tabu_tenure
        else:
            t = idx + 20
            assignment_int ^= (1 << t)
            types[t] ^= 1
            tabu_c[idx] = step + tabu_tenure

        cur_viols += best_delta

        if cur_viols < best_viols:
            best_viols = cur_viols
            best_types = list(types)
            print(f"  ** NEW BEST: {best_viols} violations at step {step+1}")

        step += 1
        if step % log_every == 0:
            elapsed_block = time.time() - t_block
            rate = log_every / max(elapsed_block, 0.001)
            t_block = time.time()
            print(f"  {step:>8d}  {cur_viols:>10d}  {best_viols:>6d}  {rate:>7.0f}/s")

    print(f"\n  Final: violations={cur_viols}  best={best_viols}")
    return best_types, best_viols


# ---------------------------------------------------------------------------
# Paley(41) construction
# ---------------------------------------------------------------------------

def quadratic_residues_mod(p):
    """Non-zero quadratic residues mod prime p."""
    return {(x*x) % p for x in range(1, p)}

def build_paley41():
    """
    Build Paley(41) as full K_42 coloring.
    Vertices 0-40: Paley(41) — edge (i,j) red iff (j-i) mod 41 is a QR.
    Vertex 41: add with greedy coloring (minimize violations from vertex 41).
    Returns 861-element assignment array.
    """
    P = 41
    QR = quadratic_residues_mod(P)

    # Build K_41 red adjacency (Paley)
    red = [[0]*N for _ in range(N)]
    for i in range(P):
        for j in range(P):
            if i != j and (j - i) % P in QR:
                red[i][j] = 1

    # Count K4s in red subgraph (needed to greedily extend vertex 41)
    # For vertex 41: try connecting to all subsets to minimize K5 violations
    # Greedy: for each edge (41, v), color red if |red K4s in N_red(41)| < |blue K4s|
    # Simpler: just assign red to the first 20 neighbors (balanced)
    # We'll greedily minimize violations from vertex 41

    print(f"  Building Paley(41)...")
    # Count violations in K_41 first
    v41_red = []  # vertices connected to 41 by red
    # Greedy: for each vertex v=0..40, decide color of edge (41,v)
    # by checking which color adds fewer K5 violations
    for v in range(P):
        # Try red: how many red K4s exist in current red neighborhood of 41 ∪ {v}?
        # red K5 from 41 = K4 in {u : 41-u is red} containing v
        cur_red_nbrs = v41_red[:]
        # If we add v to red neighbors of 41:
        # new K5s involving edge (41,v) = K4 in red graph on cur_red_nbrs ∪ {v}
        #                                 that includes v
        # = K3 in red graph on cur_red_nbrs that are ALL connected to v in red
        k4_red = 0
        for a in cur_red_nbrs:
            if not red[a][v]: continue
            for b in cur_red_nbrs:
                if b >= a: continue
                if not red[b][v] or not red[a][b]: continue
                for c in cur_red_nbrs:
                    if c >= b: continue
                    if red[c][v] and red[a][c] and red[b][c]:
                        k4_red += 1  # K4 (a,b,c,v) all red + edge to 41

        # Try blue: same but in blue (= non-red on K_41)
        blue_nbrs = [u for u in range(P) if u not in cur_red_nbrs and u != v]
        # Actually compute blue neighborhood properly
        cur_blue_nbrs = [u for u in range(P) if u not in v41_red]
        cur_blue_nbrs = [u for u in cur_blue_nbrs if u != v]
        k4_blue = 0
        for a in cur_blue_nbrs:
            if red[a][v]: continue  # a-v must be blue
            for b in cur_blue_nbrs:
                if b >= a: continue
                if red[b][v] or red[a][b]: continue
                for c in cur_blue_nbrs:
                    if c >= b: continue
                    if not red[c][v] and not red[a][c] and not red[b][c]:
                        k4_blue += 1

        if k4_red <= k4_blue:
            v41_red.append(v)
        # else: edge (41, v) is blue

    # Build assignment from red adjacency + vertex 41 connections
    red41 = set(v41_red)
    assignment = []
    for a in range(N):
        for b in range(a+1, N):
            if a < P and b < P:
                assignment.append(red[a][b])
            elif a < P and b == P:  # b = vertex 41
                assignment.append(1 if a in red41 else 0)
            else:
                assignment.append(0)  # shouldn't happen

    return assignment


# ---------------------------------------------------------------------------
# Violation counter for full assignment (Python, for verification)
# ---------------------------------------------------------------------------

def count_violations_full(assignment):
    """Count K5 violations from full 861-element assignment."""
    # Build adjacency
    adj = [[0]*N for _ in range(N)]
    idx = 0
    for a in range(N):
        for b in range(a+1, N):
            if assignment[idx]:
                adj[a][b] = adj[b][a] = 1
            idx += 1

    red_viols = 0
    for a in range(N):
        for b in range(a+1, N):
            if not adj[a][b]: continue
            for c in range(b+1, N):
                if not adj[a][c] or not adj[b][c]: continue
                for d in range(c+1, N):
                    if not adj[a][d] or not adj[b][d] or not adj[c][d]: continue
                    for e in range(d+1, N):
                        if adj[a][e] and adj[b][e] and adj[c][e] and adj[d][e]:
                            red_viols += 1

    # Blue = complement in K_42
    blue_viols = 0
    for a in range(N):
        for b in range(a+1, N):
            if adj[a][b]: continue  # skip red edges
            for c in range(b+1, N):
                if adj[a][c] or adj[b][c]: continue
                for d in range(c+1, N):
                    if adj[a][d] or adj[b][d] or adj[c][d]: continue
                    for e in range(d+1, N):
                        if not adj[a][e] and not adj[b][e] and not adj[c][e] and not adj[d][e]:
                            blue_viols += 1

    return red_viols + blue_viols


def to_checkpoint(assignment, violations=9999):
    return {'n': N, 's': 5,
            'violations': violations,
            'restart': 0,
            'total_iters': 0,
            'snap_count': 0,
            'timestamp': '2026-06-19',
            'assignment': assignment}


# ---------------------------------------------------------------------------
# Z_7 hexa-circulant: 6 groups of 7  (123 orbit types, orbit size 7)
# ---------------------------------------------------------------------------
# Vertex layout: v = pos*6 + grp  (pos in Z_7, grp in {0..5})
# Z_7 action: shift pos by +1 mod 7, keep grp fixed.
# Same-group orbit (grp=g, diff=k in {1,2,3}): type index  g*3 + (k-1)    [0..17]
# Cross-group orbit (g1<g2, diff=d in {0..6}):  type index  18 + pair*7+d  [18..122]
# ---------------------------------------------------------------------------

N7_P = 7
N7_G = 6
N7_SAME  = N7_G * 3                    # 18
N7_CPAIR = N7_G * (N7_G - 1) // 2     # 15
N7_CROSS = N7_CPAIR * N7_P             # 105
N7_TYPES = N7_SAME + N7_CROSS          # 123


def vertex_z7(v):
    return v // N7_G, v % N7_G   # (pos, grp)


def edge_orbit_z7(u, v):
    pu, gu = vertex_z7(u)
    pv, gv = vertex_z7(v)
    if gu == gv:
        diff = (pv - pu) % N7_P
        k = min(diff, N7_P - diff)   # 1..3
        return gu * 3 + (k - 1)
    else:
        if gu > gv:
            pu, pv, gu, gv = pv, pu, gv, gu
        d = (pv - pu) % N7_P
        pair_idx = gu * (2 * N7_G - gu - 1) // 2 + (gv - gu - 1)
        return N7_SAME + pair_idx * N7_P + d


_U1 = np.uint64(1)   # typed 1 for uint64 shifts

def build_lookup_tables_z7(verbose=True):
    """
    Returns (mults_np, masks_lo, masks_hi, affected_np).
    masks_lo[i] = bits 0..63 of the 123-bit type mask for canonical tuple i.
    masks_hi[i] = bits 64..122 (stored in bits 0..58 of the high word).
    """
    if verbose:
        print("  [Z7] Precomputing 5-tuple lookup tables... ", end="", flush=True)
    t0 = time.time()

    rot_z7 = [[(v // N7_G + k) % N7_P * N7_G + v % N7_G for k in range(N7_P)]
               for v in range(N)]

    canon_map = {}
    for combo in itertools.combinations(range(N), 5):
        best = None
        for v in combo:
            shift = (N7_P - v // N7_G) % N7_P
            candidate = tuple(sorted(rot_z7[w][shift] for w in combo))
            if best is None or candidate < best:
                best = candidate
        if best in canon_map:
            canon_map[best][0] += 1
        else:
            etypes = [edge_orbit_z7(best[i], best[j])
                      for i in range(5) for j in range(i + 1, 5)]
            canon_map[best] = [1, etypes]

    canon_list = list(canon_map.values())
    n_orbits   = len(canon_list)
    mults_np   = np.array([c[0] for c in canon_list], dtype=np.int64)

    masks_lo = np.zeros(n_orbits, dtype=np.uint64)
    masks_hi = np.zeros(n_orbits, dtype=np.uint64)
    for idx, (_, etypes) in enumerate(canon_list):
        for t in set(etypes):
            if t < 64:
                masks_lo[idx] |= _U1 << np.uint64(t)
            else:
                masks_hi[idx] |= _U1 << np.uint64(t - 64)

    affected = [[] for _ in range(N7_TYPES)]
    for idx, (_, etypes) in enumerate(canon_list):
        for t in set(etypes):
            affected[t].append(idx)
    affected_np = [np.array(a, dtype=np.int32) for a in affected]

    elapsed = time.time() - t0
    if verbose:
        avg = sum(len(a) for a in affected_np) / N7_TYPES
        print(f"done in {elapsed:.1f}s  ({n_orbits} canonical 5-tuples, avg {avg:.0f} per type)")
    return mults_np, masks_lo, masks_hi, affected_np


def _types_to_lohi(types):
    lo, hi = np.uint64(0), np.uint64(0)
    for t, c in enumerate(types):
        if c:
            if t < 64:
                lo |= _U1 << np.uint64(t)
            else:
                hi |= _U1 << np.uint64(t - 64)
    return lo, hi


def count_violations_z7(lo, hi, masks_lo, masks_hi, mults_np):
    mlo = masks_lo & lo
    mhi = masks_hi & hi
    mono = ((mlo == np.uint64(0)) & (mhi == np.uint64(0))) | \
           ((mlo == masks_lo) & (mhi == masks_hi))
    return int(np.sum(mults_np[mono]))


def delta_violations_z7(lo, hi, flip_t, masks_lo, masks_hi, mults_np, affected_np):
    idxs   = affected_np[flip_t]
    slo    = masks_lo[idxs]
    shi    = masks_hi[idxs]
    smult  = mults_np[idxs]

    if flip_t < 64:
        nlo, nhi = lo ^ (_U1 << np.uint64(flip_t)), hi
    else:
        nlo, nhi = lo, hi ^ (_U1 << np.uint64(flip_t - 64))

    mlo_b = slo & lo;    mhi_b = shi & hi
    mlo_a = slo & nlo;   mhi_a = shi & nhi
    U0 = np.uint64(0)
    mono_b = ((mlo_b == U0) & (mhi_b == U0)) | ((mlo_b == slo) & (mhi_b == shi))
    mono_a = ((mlo_a == U0) & (mhi_a == U0)) | ((mlo_a == slo) & (mhi_a == shi))
    return int(np.dot((mono_a.astype(np.int64) - mono_b.astype(np.int64)), smult))


def assignment_from_types_z7(types_z7):
    assignment = [0] * (N * (N - 1) // 2)
    idx = 0
    for a in range(N):
        for b in range(a + 1, N):
            assignment[idx] = int(types_z7[edge_orbit_z7(a, b)])
            idx += 1
    return assignment


def tabu_search_z7(types, mults_np, masks_lo, masks_hi, affected_np,
                   steps, tabu_tenure, rng, log_every=1000, verbose=True):
    lo, hi     = _types_to_lohi(types)
    best_types = list(types)
    best_viols = count_violations_z7(lo, hi, masks_lo, masks_hi, mults_np)
    cur_viols  = best_viols

    tabu = [-tabu_tenure] * N7_TYPES
    step = 0

    if verbose:
        print(f"  [Z7] Start: {best_viols}v  ({N7_TYPES} types: {N7_SAME} same-grp + {N7_CROSS} cross-grp)")
        print(f"  {'step':>8}  {'violations':>10}  {'best':>6}  {'steps/s':>8}")
        print("  " + "-"*42)
    t_block = time.time()

    while step < steps:
        best_delta = None
        best_t     = -1
        for t in range(N7_TYPES):
            is_tabu = step < tabu[t]
            d = delta_violations_z7(lo, hi, t, masks_lo, masks_hi, mults_np, affected_np)
            aspiration = (cur_viols + d < best_viols)
            if is_tabu and not aspiration:
                continue
            if best_delta is None or d < best_delta:
                best_delta = d
                best_t     = t

        if best_t == -1:
            best_t     = rng.randrange(N7_TYPES)
            best_delta = delta_violations_z7(lo, hi, best_t, masks_lo, masks_hi, mults_np, affected_np)

        # Apply flip
        types[best_t] ^= 1
        if best_t < 64:
            lo ^= _U1 << np.uint64(best_t)
        else:
            hi ^= _U1 << np.uint64(best_t - 64)
        cur_viols += best_delta
        tabu[best_t] = step + tabu_tenure
        step += 1

        if cur_viols < best_viols:
            best_viols = cur_viols
            best_types = list(types)
            if verbose:
                print(f"  ** NEW BEST: {best_viols} violations at step {step}")

        if verbose and step % log_every == 0:
            elapsed_b = time.time() - t_block
            rate = log_every / max(elapsed_b, 0.001)
            t_block = time.time()
            print(f"  {step:>8d}  {cur_viols:>10d}  {best_viols:>6d}  {rate:>7.0f}/s")

    if verbose:
        print(f"\n  Final: violations={cur_viols}  best={best_viols}")
    return best_types, best_viols


# ---------------------------------------------------------------------------
# K_43 BI-CIRCULANT SEARCH
# ---------------------------------------------------------------------------
# K_43 = Z_21 x {0,1} + one extra vertex (index 42).
# Orbit decomposition under Z_21 action (vertex 42 is fixed):
#   Types 0-9  : A_k within half-0  (same as K_42)
#   Types 10-19: B_k within half-1  (same as K_42)
#   Types 20-40: C_k between halves (same as K_42)
#   Type 41    : D0 - all 21 edges from vertex 42 to half-0
#   Type 42    : D1 - all 21 edges from vertex 42 to half-1
# Total: 43 orbit types x 21 edges each = 903 edges = C(43,2). Search: 2^43.
# Since 43 < 64, we use the same single Python int / int64 bitmask as K_42.

N43       = 43
N43_TYPES = 43
N43_EXTRA = 42   # index of the extra vertex


def edge_orbit_k43(u, v):
    """Orbit type for K_43 bi-circulant edge (u,v), u < v. Range [0, 42]."""
    if u > v:
        u, v = v, u
    if v == N43_EXTRA:           # one endpoint is the extra vertex; other is u
        return 41 if u < N1 else 42   # D0 (half-0) or D1 (half-1)
    return edge_orbit(u, v)      # types 0-40: identical to K_42


def build_lookup_tables_k43(verbose=True, _k42_tables=None):
    """
    Build violation-counting tables for K_43 bi-circulant search.

    Strategy: reuse the K_42 canonical orbits verbatim, then enumerate all
    C(42,4) = 111,930 four-tuples from {0..41} and combine each with vertex
    42 to get the extra K_5 family.  Canonical orbit under Z_21 is the same
    rotation trick as K_42 (vertex 42 is fixed, so it's always in the tuple).

    Returns (masks_np, mults_np, type_to_canons_np) — shapes:
      masks_np : (n_orbits,) int64   bitmask of orbit types for each canonical K_5
      mults_np : (n_orbits,) int64   orbit size (multiplicity)
      type_to_canons_np : list of 43 int32 arrays (indices into combined array)
    """
    # --- K_42 part ---
    if _k42_tables is not None:
        _, masks_42, mults_42, _, _ = _k42_tables
    else:
        if verbose:
            print("  [K43] Building K_42 sub-tables (reuse)...")
        _, masks_42, mults_42, _, _ = build_lookup_tables(verbose=verbose)
    n_42_orbits = len(masks_42)

    # --- K_43 extension: 5-tuples through vertex 42 ---
    if verbose:
        print("  [K43] Enumerating C(42,4)=111,930 four-tuples for vertex-42 extension... ",
              end="", flush=True)
    t0 = time.time()

    rot_vtx = [[(v % N1 + k) % N1 + (v // N1) * N1 for k in range(N1)]
               for v in range(N)]   # N=42

    canon_map_43 = {}  # canonical 4-tuple key -> [multiplicity, [10 edge types]]

    for combo in itertools.combinations(range(N), 4):   # 4 vertices from 0..41
        # Canonical rotation: try each vertex's shift-to-0
        best = None
        for v in combo:
            shift = (N1 - v % N1) % N1
            candidate = tuple(sorted(rot_vtx[w][shift] for w in combo))
            if best is None or candidate < best:
                best = candidate

        if best in canon_map_43:
            canon_map_43[best][0] += 1
        else:
            # 5-tuple = best + (42,); compute 10 edge types
            five = list(best) + [N43_EXTRA]
            etypes = [edge_orbit_k43(five[i], five[j])
                      for i in range(5) for j in range(i + 1, 5)]
            canon_map_43[best] = [1, etypes]

    canon_list_43 = list(canon_map_43.values())
    n_43_ext = len(canon_list_43)

    masks_43 = np.zeros(n_43_ext, dtype=np.int64)
    mults_43 = np.zeros(n_43_ext, dtype=np.int64)
    for idx, (mult, etypes) in enumerate(canon_list_43):
        mults_43[idx] = mult
        m = 0
        for t in etypes:
            m |= (1 << t)
        masks_43[idx] = m

    elapsed = time.time() - t0
    if verbose:
        print(f"done in {elapsed:.1f}s  ({n_43_ext} canonical 5-tuples through v42)")

    # --- Combine ---
    all_masks = np.concatenate([masks_42, masks_43])
    all_mults = np.concatenate([mults_42, mults_43])

    # type_to_canons for all 43 types
    type_to_canons_k43 = [[] for _ in range(N43_TYPES)]
    for idx in range(n_42_orbits):
        m = int(masks_42[idx])
        for t in range(41):          # only types 0-40 appear in K_42 part
            if m & (1 << t):
                type_to_canons_k43[t].append(idx)
    for idx, (mult, etypes) in enumerate(canon_list_43):
        for t in set(etypes):
            type_to_canons_k43[t].append(n_42_orbits + idx)

    type_to_canons_k43_np = [np.array(tc, dtype=np.int32)
                              for tc in type_to_canons_k43]

    total = n_42_orbits + n_43_ext
    if verbose:
        print(f"  [K43] Combined: {n_42_orbits} K_42 + {n_43_ext} K_43-ext = {total} canonical K_5 orbits")

    return all_masks, all_mults, type_to_canons_k43_np


def count_violations_k43(assignment_int, masks_np, mults_np):
    """Count monochromatic K_5 violations in K_43 bi-circulant coloring."""
    masked = masks_np & assignment_int
    is_mono = (masked == 0) | (masked == masks_np)
    return int(np.sum(mults_np[is_mono]))


def delta_violations_k43(assignment_int, flip_t, masks_np, mults_np,
                          type_to_canons_np):
    """Change in violation count if orbit type flip_t is flipped."""
    cands = type_to_canons_np[flip_t]
    if len(cands) == 0:
        return 0
    sub_masks = masks_np[cands]
    sub_mults = mults_np[cands]
    masked_old = sub_masks & assignment_int
    is_mono_old = (masked_old == 0) | (masked_old == sub_masks)
    new_int = assignment_int ^ (1 << flip_t)
    masked_new = sub_masks & new_int
    is_mono_new = (masked_new == 0) | (masked_new == sub_masks)
    return int(np.sum(sub_mults[is_mono_new]) - np.sum(sub_mults[is_mono_old]))


def assignment_from_types_k43(types_k43):
    """Convert 43-type K_43 bi-circulant coloring to full 903-edge assignment."""
    n_edges = N43 * (N43 - 1) // 2
    assignment = [0] * n_edges
    idx = 0
    for a in range(N43):
        for b in range(a + 1, N43):
            t = edge_orbit_k43(a, b)
            assignment[idx] = int(types_k43[t])
            idx += 1
    return assignment


def tabu_search_k43(types, masks_np, mults_np, type_to_canons_np,
                    steps, tabu_tenure, rng, log_every=1000, verbose=True):
    """K_43 bi-circulant tabu search. Returns (best_types, best_viols)."""
    assignment_int = sum(int(types[t]) << t for t in range(N43_TYPES))
    violations = count_violations_k43(assignment_int, masks_np, mults_np)
    best_types = types[:]
    best_viols = violations

    tabu = [-tabu_tenure] * N43_TYPES
    t0 = time.time()

    if verbose:
        print(f"  [K43] Start: {violations}v  (43 orbit types: 41 K_42 + D0 + D1)")
        print(f"  {'step':>8}  {'violations':>10}  {'best':>6}  {'steps/s':>8}")
        print("  " + "-"*44)
    t_block = time.time()

    for step in range(steps):
        best_delta = None
        best_t = -1

        for flip_t in range(N43_TYPES):
            is_tabu = step < tabu[flip_t]
            d = delta_violations_k43(assignment_int, flip_t, masks_np,
                                     mults_np, type_to_canons_np)
            aspiration = (violations + d < best_viols)
            if is_tabu and not aspiration:
                continue
            if best_delta is None or d < best_delta:
                best_delta = d
                best_t = flip_t

        if best_t == -1:
            best_t = rng.randrange(N43_TYPES)
            best_delta = delta_violations_k43(assignment_int, best_t,
                                              masks_np, mults_np, type_to_canons_np)

        tabu[best_t] = step + tabu_tenure
        assignment_int ^= (1 << best_t)
        types[best_t] ^= 1
        violations += best_delta

        if violations < best_viols:
            best_viols = violations
            best_types = types[:]
            if verbose:
                print(f"  ** NEW BEST: {best_viols}v at step {step+1}")
            if best_viols == 0:
                if verbose:
                    print("  *** VALID K_43 COLORING FOUND! R(5,5) >= 44 PROVEN! ***")
                break

        if verbose and (step + 1) % log_every == 0:
            elapsed_b = time.time() - t_block
            rate = log_every / max(elapsed_b, 0.001)
            t_block = time.time()
            print(f"  {step+1:>8d}  {violations:>10d}  {best_viols:>6d}  {rate:>7.0f}/s")

    if verbose:
        elapsed = time.time() - t0
        print(f"\n  Final: violations={violations}  best={best_viols}  ({elapsed:.1f}s)")

    return best_types, best_viols


def tabu_search_k43_fixed(types, fixed_mask_bits, masks_np, mults_np, type_to_canons_np,
                          steps, tabu_tenure, rng, log_every=1000, verbose=True):
    """
    K_43 bi-circulant tabu search with some orbit types FIXED.
    fixed_mask_bits: integer bitmask of which type indices are frozen (cannot be flipped).
    Returns (best_types, best_viols).
    """
    assignment_int = sum(int(types[t]) << t for t in range(N43_TYPES))
    violations = count_violations_k43(assignment_int, masks_np, mults_np)
    best_types = types[:]
    best_viols = violations

    tabu = [-tabu_tenure] * N43_TYPES
    free_types = [t for t in range(N43_TYPES) if not (fixed_mask_bits >> t & 1)]
    t0 = time.time()
    t_block = time.time()

    if verbose:
        fixed_indices = [t for t in range(N43_TYPES) if (fixed_mask_bits >> t & 1)]
        fixed_values = {t: types[t] for t in fixed_indices}
        print(f"  [K43-FIXED] Start: {violations}v  fixed={fixed_values}  free={len(free_types)} types")

    for step in range(steps):
        best_delta = None
        best_t = -1

        for flip_t in free_types:
            is_tabu = step < tabu[flip_t]
            d = delta_violations_k43(assignment_int, flip_t, masks_np, mults_np, type_to_canons_np)
            aspiration = (violations + d < best_viols)
            if is_tabu and not aspiration:
                continue
            if best_delta is None or d < best_delta:
                best_delta = d
                best_t = flip_t

        if best_t == -1:
            best_t = rng.choice(free_types)
            best_delta = delta_violations_k43(assignment_int, best_t, masks_np, mults_np, type_to_canons_np)

        tabu[best_t] = step + tabu_tenure
        assignment_int ^= (1 << best_t)
        types[best_t] ^= 1
        violations += best_delta

        if violations < best_viols:
            best_viols = violations
            best_types = types[:]
            if verbose:
                print(f"    ** NEW BEST: {best_viols}v at step {step+1}")
            if best_viols == 0:
                print("  *** VALID K_43 COLORING FOUND! R(5,5) >= 44 PROVEN! ***")
                break

        if verbose and (step + 1) % log_every == 0:
            elapsed_b = time.time() - t_block
            rate = log_every / max(elapsed_b, 0.001)
            t_block = time.time()
            print(f"  {step+1:>8d}  {violations:>10d}  {best_viols:>6d}  {rate:>7.0f}/s")

    elapsed = time.time() - t0
    if verbose:
        print(f"  Final: {best_viols}v  ({elapsed:.1f}s)")
    return best_types, best_viols


def k43_irdme_loop(masks_np, mults_np, type_to_canons_k43, args, rng):
    """
    IRDME loop for K_43:
      Atlas finding: type 41 (D0, v*→half-0) and type 42 (D1, v*→half-1)
      are the orbit types for the extra vertex v42.
      The IRDME analysis showed v42 is the top violation hub.
      Strategy: enumerate all 4 (D0, D1) combinations, fix them,
      run tabu on the remaining 41 orbit types, compare results.
    """
    print("\n" + "="*65)
    print("  K_43 IRDME Loop — D0/D1 Orbit Enumeration")
    print("="*65)
    print("  IRDME atlas finding: v42 (extra vertex) = top violation hub")
    print("  D0 = orbit type 41: v42 -> half-0 (21 edges)")
    print("  D1 = orbit type 42: v42 -> half-1 (21 edges)")
    print("  Enumerating all 4 (D0,D1) in {0,1}^2 with remaining 41 types free")
    print()

    # Load warm-start from checkpoint if available
    if args.from_checkpoint:
        ckpt = json.load(open(args.from_checkpoint))
        asgn = ckpt['assignment']
        base_types = [0] * N43_TYPES
        if len(asgn) == 861:
            idx = 0
            for a in range(N):
                for b in range(a + 1, N):
                    base_types[edge_orbit(a, b)] = asgn[idx]
                    idx += 1
        else:
            idx = 0
            for a in range(N43):
                for b in range(a + 1, N43):
                    base_types[edge_orbit_k43(a, b)] = asgn[idx]
                    idx += 1
        base_ai = sum(int(base_types[t]) << t for t in range(N43_TYPES))
        base_v = count_violations_k43(base_ai, masks_np, mults_np)
        print(f"  Warm-start checkpoint: {base_v}v")
        print(f"  Existing D0={base_types[41]}  D1={base_types[42]}")
    else:
        base_types = None
        base_v = None
        print("  No checkpoint — using random starts for each trial")
    print()

    # Fixed bitmask = bits 41 and 42
    fixed_mask_bits = (1 << 41) | (1 << 42)

    global_best_viols = base_v if base_v is not None else 999999
    global_best_types = base_types[:] if base_types is not None else None
    results = []

    for d0, d1 in [(0,0),(0,1),(1,0),(1,1)]:
        print(f"\n{'-'*55}")
        label = {(0,0):'blue->h0, blue->h1',(0,1):'blue->h0, red->h1',(1,0):'red->h0, blue->h1',(1,1):'red->h0, red->h1'}[(d0,d1)]
        print(f"  Trial (D0={d0}, D1={d1}): {label}")
        print(f"{'-'*55}")

        # Build starting types: warm-start (or random) + forced D0/D1
        if base_types is not None:
            types = base_types[:]
        else:
            types = [rng.randint(0, 1) for _ in range(N43_TYPES)]
        types[41] = d0
        types[42] = d1

        best_t, best_v = tabu_search_k43_fixed(
            types, fixed_mask_bits, masks_np, mults_np, type_to_canons_k43,
            args.steps, args.tabu, rng, args.log_every
        )
        results.append((d0, d1, best_v, best_t))
        print(f"  Trial (D0={d0},D1={d1}) result: {best_v}v")

        if best_v < global_best_viols:
            global_best_viols = best_v
            global_best_types = best_t[:]
            print(f"  ** New global best: {global_best_viols}v")

    print(f"\n{'='*55}")
    print(f"  IRDME Loop Summary:")
    print(f"  {'D0':>4}  {'D1':>4}  {'violations':>12}")
    for d0, d1, v, _ in sorted(results, key=lambda x: x[2]):
        marker = " <- best" if v == global_best_viols else ""
        print(f"  {d0:>4}  {d1:>4}  {v:>12}{marker}")
    print(f"\n  Global best: {global_best_viols}v")
    print(f"{'='*55}")

    if global_best_types is not None:
        assignment = assignment_from_types_k43(global_best_types)
        out = Path(args.out)
        ckpt_out = {
            'n': N43, 's': 5, 'violations': global_best_viols,
            'restart': 0, 'total_iters': 0, 'snap_count': 0,
            'timestamp': '2026-06-20', 'assignment': assignment
        }
        with open(out, "w") as f:
            json.dump(ckpt_out, f)
        print(f"\n  Saved best -> {out}")
        if global_best_viols == 0:
            sol_path = str(out).replace('.json', '_SOLUTION.json')
            with open(sol_path, "w") as f:
                json.dump(ckpt_out, f)
            print(f"  *** SOLUTION SAVED -> {sol_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_84V = [
    "datasets/RAMSEY_K42_S5_bicirculant_84v.json",
    "datasets/RAMSEY_K42_S5_bicirculant_84v_s99.json",
    "datasets/RAMSEY_K42_S5_bicirculant_84v_hi_s42.json",
]

def crossover_search(cols, masks_np, mults_np, type_to_canons_np,
                     steps_per, tabu_tenure, rng, n_random,
                     log_every=500):
    """
    Population crossover from 3 parent colorings.
    Phase 1: all 27 block-level hybrids (A from i, B from j, C from k).
    Phase 2: n_random bit-level hybrids (each bit drawn from random parent).
    Returns best (types, viols) found across all candidates.
    """
    best_viols = 9999
    best_types = None

    candidates = []

    # --- 27 block-level hybrids ---
    print("  Phase 1: 27 block-level hybrids  (A_i | B_j | C_k)")
    print("  " + "-"*55)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                types = cols[i][0:10] + cols[j][10:20] + cols[k][20:41]
                cand_int = types_to_int(types)
                init_v = count_violations_np(cand_int, masks_np, mults_np)
                candidates.append(('block', (i, j, k), list(types), init_v))

    # Sort by initial violations so we run promising ones first
    candidates.sort(key=lambda x: x[3])
    for ci, (ctype, key, types, init_v) in enumerate(candidates):
        i, j, k = key
        same_as_parent = (i == j == k)
        label = "parent" if same_as_parent else "hybrid"
        print(f"  [{ci+1:2d}/27] A{i+1}|B{j+1}|C{k+1} ({label}): init={init_v:4d}v", end="", flush=True)
        t_types, t_viols = tabu_search_bicirculant(
            list(types), masks_np, mults_np, type_to_canons_np,
            steps_per, tabu_tenure, rng, log_every=log_every, verbose=False
        )
        print(f" -> best={t_viols}v")
        if t_viols < best_viols:
            best_viols = t_viols
            best_types = list(t_types)
            print(f"  *** CROSSOVER BEST: {best_viols}v  (A{i+1}|B{j+1}|C{k+1})")
        if best_viols == 0:
            return best_types, best_viols

    # --- Phase 2: random bit-level hybrids ---
    if n_random > 0:
        print()
        print(f"  Phase 2: {n_random} random bit-level hybrids")
        print("  " + "-"*55)
        for r in range(n_random):
            types = [cols[rng.randrange(len(cols))][t] for t in range(N_TYPES)]
            cand_int = types_to_int(types)
            init_v = count_violations_np(cand_int, masks_np, mults_np)
            print(f"  [rnd {r+1:3d}/{n_random}]: init={init_v:4d}v", end="", flush=True)
            t_types, t_viols = tabu_search_bicirculant(
                list(types), masks_np, mults_np, type_to_canons_np,
                steps_per, tabu_tenure, rng, log_every=log_every, verbose=False
            )
            print(f" -> best={t_viols}v")
            if t_viols < best_viols:
                best_viols = t_viols
                best_types = list(t_types)
                print(f"  *** CROSSOVER BEST: {best_viols}v  (random bit-cross #{r+1})")
            if best_viols == 0:
                return best_types, best_viols

    return best_types, best_viols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bicirculant","sc","hi","paley","both","crossover","z7","k43","k43_irdme"],
                    default="bicirculant")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--tabu", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=5000)
    ap.add_argument("--out", default="datasets/ramsey_symmetry_result.json")
    ap.add_argument("--from-checkpoint", default=None, metavar="FILE",
                    help="Load starting types from a saved JSON coloring")
    ap.add_argument("--verify", action="store_true",
                    help="Verify violation count with full Python counter (slow ~10s)")
    # crossover-specific
    ap.add_argument("--steps-per", type=int, default=2000,
                    help="Tabu steps per candidate in crossover mode (default 2000)")
    ap.add_argument("--random-crosses", type=int, default=50,
                    help="Number of random bit-level hybrids in crossover mode (default 50)")
    ap.add_argument("--coloring1", default=DEFAULT_84V[0], metavar="FILE")
    ap.add_argument("--coloring2", default=DEFAULT_84V[1], metavar="FILE")
    ap.add_argument("--coloring3", default=DEFAULT_84V[2], metavar="FILE")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("\n" + "="*65)
    print("  Symmetry-Restricted Ramsey Search  --  Z_21 Bi-Circulant K_42")
    print("="*65)
    print(f"  41 orbit types x 21 edges each = 861 total edges")
    print(f"  Search space: 2^41 = {2**41/1e12:.1f}T colorings")
    print()

    if args.mode == "paley":
        print("  [PALEY MODE] Building Paley(41) + vertex-41 extension")
        print("  " + "-"*50)
        assignment = build_paley41()
        t0 = time.time()
        print(f"  Counting violations... ", end="", flush=True)
        viols = count_violations_full(assignment)
        print(f"{viols}  ({time.time()-t0:.1f}s)")
        out = Path(args.out).with_suffix(".paley.json")
        with open(out, "w") as f:
            json.dump(to_checkpoint(assignment, viols), f)
        print(f"  Saved -> {out}")
        print()
        print("  To continue from this starting point:")
        print(f"    Copy {out} to datasets\\RAMSEY_K42_S5_rust_checkpoint.json")
        print(f"    rust\\ramsey\\target\\release\\ramsey.exe --n 42 --s 5 --iter 200000 --restarts 500 --tabu 12 --plateau 20000")
        return

    # --- BI-CIRCULANT / SC MODE ---

    # Precompute
    canon_list, masks_np, mults_np, type_to_canons_np, sc_a_canons_np = build_lookup_tables(verbose=True)
    avg_sz = sum(len(x) for x in type_to_canons_np) / N_TYPES
    print(f"  Type list sizes: avg {avg_sz:.0f} tuples/type")
    print()

    # Starting coloring: from checkpoint or random
    if args.from_checkpoint:
        ckpt = json.load(open(args.from_checkpoint))
        types = types_from_assignment(ckpt['assignment'])
        init_int   = types_to_int(types)
        init_viols = count_violations_np(init_int, masks_np, mults_np)
        print(f"  Checkpoint start: {init_viols} violations  ({args.from_checkpoint})")
    else:
        types = [rng.randint(0, 1) for _ in range(N_TYPES)]
        init_int   = types_to_int(types)
        init_viols = count_violations_np(init_int, masks_np, mults_np)
        print(f"  Random start: {init_viols} violations")
    print()

    # --- K43 BI-CIRCULANT MODE ---
    if args.mode == "k43":
        print("\n" + "="*65)
        print("  K_43 Bi-Circulant Search  (Z_21 + extra vertex)")
        print("="*65)
        print(f"  43 orbit types x 21 edges each = 903 total edges = C(43,2)")
        print(f"  Search space: 2^43 = {2**43/1e12:.1f}T colorings")
        print(f"  ** If best reaches 0: R(5,5) >= 44 (NEW MATHEMATICAL RESULT) **")
        print()
        all_masks, all_mults, type_to_canons_k43 = build_lookup_tables_k43(
            verbose=True, _k42_tables=(None, masks_np, mults_np, None, None))
        print()

        if args.from_checkpoint:
            ckpt  = json.load(open(args.from_checkpoint))
            # Checkpoint may be K_42 (861 edges) or K_43 (903 edges)
            asgn = ckpt['assignment']
            if len(asgn) == 861:    # K_42 coloring -> embed into K_43, color new edges randomly
                print(f"  K_42 checkpoint detected ({len(asgn)} edges) -> embedding into K_43")
                types = [0] * N43_TYPES
                idx = 0
                for a in range(N):
                    for b in range(a + 1, N):
                        types[edge_orbit(a, b)] = asgn[idx]
                        idx += 1
                types[41] = rng.randint(0, 1)   # D0
                types[42] = rng.randint(0, 1)   # D1
            else:                                # Full K_43 checkpoint
                types = [0] * N43_TYPES
                idx = 0
                for a in range(N43):
                    for b in range(a + 1, N43):
                        types[edge_orbit_k43(a, b)] = asgn[idx]
                        idx += 1
            init_ai = sum(int(types[t]) << t for t in range(N43_TYPES))
            init_v  = count_violations_k43(init_ai, all_masks, all_mults)
            print(f"  Checkpoint start: {init_v}v")
        else:
            types = [rng.randint(0, 1) for _ in range(N43_TYPES)]
            init_ai = sum(int(types[t]) << t for t in range(N43_TYPES))
            init_v  = count_violations_k43(init_ai, all_masks, all_mults)
            print(f"  Random start: {init_v}v")
        print()

        t0 = time.time()
        best_types, best_viols = tabu_search_k43(
            types, all_masks, all_mults, type_to_canons_k43,
            args.steps, args.tabu, rng, args.log_every
        )
        elapsed = time.time() - t0
        rate = int(args.steps / max(elapsed, 0.001))
        print(f"  Elapsed: {elapsed:.1f}s  ({rate} steps/sec)")

        # Save result in Rust-compatible checkpoint format (n=43)
        assignment = assignment_from_types_k43(best_types)
        out = Path(args.out)
        ckpt_out = {
            'n': N43, 's': 5, 'violations': best_viols,
            'restart': 0, 'total_iters': 0, 'snap_count': 0,
            'timestamp': '2026-06-20', 'assignment': assignment
        }
        with open(out, "w") as f:
            json.dump(ckpt_out, f)
        print(f"  Saved -> {out}  (violations={best_viols})")
        if best_viols == 0:
            sol_path = str(out).replace('.json', '_SOLUTION.json')
            with open(sol_path, "w") as f:
                json.dump(ckpt_out, f)
            print(f"  *** SOLUTION ALSO SAVED -> {sol_path}")
        return

    # --- K43 IRDME LOOP MODE ---
    if args.mode == "k43_irdme":
        all_masks, all_mults, type_to_canons_k43 = build_lookup_tables_k43(
            verbose=True, _k42_tables=(None, masks_np, mults_np, None, None))
        print()
        k43_irdme_loop(all_masks, all_mults, type_to_canons_k43, args, rng)
        return

    # --- Z7 MODE ---
    if args.mode == "z7":
        print("  [Z7 MODE] Z_7 hexa-circulant: 6 groups of 7 vertices")
        print("  Orbit structure: 18 same-group + 105 cross-group = 123 types (orbit size 7)")
        print(f"  Search space: 2^123 = {2**123:.2e} colorings")
        print()
        mults_z7, masks_lo_z7, masks_hi_z7, affected_z7 = build_lookup_tables_z7(verbose=True)
        print()

        if args.from_checkpoint:
            ckpt  = json.load(open(args.from_checkpoint))
            types = [0] * N7_TYPES
            idx   = 0
            for a in range(N):
                for b in range(a + 1, N):
                    t = edge_orbit_z7(a, b)
                    types[t] = ckpt['assignment'][idx]
                    idx += 1
            lo0, hi0 = _types_to_lohi(types)
            init_v   = count_violations_z7(lo0, hi0, masks_lo_z7, masks_hi_z7, mults_z7)
            print(f"  Checkpoint start: {init_v}v")
        else:
            types  = [rng.randint(0, 1) for _ in range(N7_TYPES)]
            lo0, hi0 = _types_to_lohi(types)
            init_v   = count_violations_z7(lo0, hi0, masks_lo_z7, masks_hi_z7, mults_z7)
            print(f"  Random start: {init_v}v")
        print()

        t0 = time.time()
        best_types, best_viols = tabu_search_z7(
            types, mults_z7, masks_lo_z7, masks_hi_z7, affected_z7,
            args.steps, args.tabu, rng, args.log_every
        )
        elapsed = time.time() - t0
        rate    = int(args.steps / max(elapsed, 0.001))
        print(f"  Elapsed: {elapsed:.1f}s  ({rate} steps/sec)")

        assignment = assignment_from_types_z7(best_types)
        out = Path(args.out)
        with open(out, "w") as f:
            json.dump(to_checkpoint(assignment, best_viols), f)
        print(f"  Saved -> {out}  (violations={best_viols})")
        return

    # --- CROSSOVER MODE ---
    if args.mode == "crossover":
        print("  [CROSSOVER MODE] Loading 3 parent colorings...")
        parent_files = [args.coloring1, args.coloring2, args.coloring3]
        cols = []
        for pf in parent_files:
            d = json.load(open(pf))
            t = types_from_assignment(d['assignment'])
            v = count_violations_np(types_to_int(t), masks_np, mults_np)
            print(f"    {pf}: {v}v")
            cols.append(t)
        print()
        print(f"  steps_per={args.steps_per}  random_crosses={args.random_crosses}  tabu={args.tabu}")
        print()
        t0 = time.time()
        best_types, best_viols = crossover_search(
            cols, masks_np, mults_np, type_to_canons_np,
            args.steps_per, args.tabu, rng, args.random_crosses,
            log_every=max(100, args.steps_per // 10)
        )
        elapsed = time.time() - t0
        print(f"\n  Crossover search done: best={best_viols}v  elapsed={elapsed:.1f}s")
        assignment = assignment_from_types(best_types)
        out = Path(args.out)
        with open(out, "w") as f:
            json.dump(to_checkpoint(assignment, best_viols), f)
        print(f"  Saved -> {out}")
        return

    # Run search
    if args.mode in ("sc", "hi"):
        hi = (args.mode == "hi")
        label = "Half-identical" if hi else "Self-complementary"
        constraint = "B_k = A_k" if hi else "B_k = 1-A_k"
        print(f"  {label} tabu: {args.steps} steps, tenure={args.tabu}")
        print(f"  ({constraint} forced; 31 free bits)")
        print()
        t0 = time.time()
        best_types, best_viols = tabu_search_sc(
            types, masks_np, mults_np, type_to_canons_np, sc_a_canons_np,
            args.steps, args.tabu, rng, args.log_every, hi_mode=hi
        )
    else:
        print(f"  Tabu search: {args.steps} steps, tenure={args.tabu}")
        print()
        t0 = time.time()
        best_types, best_viols = tabu_search_bicirculant(
            types, masks_np, mults_np, type_to_canons_np,
            args.steps, args.tabu, rng, args.log_every
        )
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s  ({int(args.steps/elapsed)} steps/sec)")

    # Convert best to full assignment
    assignment = assignment_from_types(best_types)

    if args.verify:
        print(f"\n  Verifying with full K5 counter... ", end="", flush=True)
        t1 = time.time()
        full_viols = count_violations_full(assignment)
        print(f"{full_viols}  ({time.time()-t1:.1f}s)")
        if full_viols != best_viols:
            print(f"  WARNING: bicirculant count={best_viols}, full count={full_viols}")
            print(f"  Difference likely due to orbit multiplicity rounding.")
        best_viols = full_viols

    out = Path(args.out)
    with open(out, "w") as f:
        json.dump(to_checkpoint(assignment, best_viols), f)
    print(f"\n  Saved -> {out}  (violations={best_viols})")
    print()
    print("  Best type assignment:")
    type_names = (
        [f"A{k+1:02d}" for k in range(10)] +
        [f"B{k+1:02d}" for k in range(10)] +
        [f"C{k:02d}"  for k in range(21)]
    )
    for i, (name, c) in enumerate(zip(type_names, best_types)):
        print(f"    {name}={'R' if c else 'B'}", end="  ")
        if (i+1) % 10 == 0: print()
    print()
    print("  To continue with Rust local search:")
    print(f"    copy {out} datasets\\RAMSEY_K42_S5_rust_checkpoint.json")
    print(f"    rust\\ramsey\\target\\release\\ramsey.exe --n 42 --s 5 --iter 200000 --restarts 500 --tabu 12 --plateau 20000")

    if args.mode in ("both",):
        print("\n  [PALEY MODE] Running Paley extension for comparison...")
        assignment_p = build_paley41()
        viols_p = count_violations_full(assignment_p)
        print(f"  Paley(41) extended: {viols_p} violations")
        out_p = Path(args.out).with_suffix(".paley.json")
        with open(out_p, "w") as f:
            json.dump(to_checkpoint(assignment_p, viols_p), f)
        print(f"  Paley saved -> {out_p}")


if __name__ == "__main__":
    main()
