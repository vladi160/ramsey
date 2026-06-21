"""
IRDME-guided Ramsey search.

Instead of minimizing a scalar violation count, this uses the full IRDME
dimensional analysis to understand the structural health of a coloring and
guide edge flips.

The approach:
  1. Build a Ramsey coloring (structured Paley+FPL seed by default)
  2. Compute vertex dimensions: red_degree, blue_degree, red_triangles,
     blue_triangles, red_violations, blue_violations
  3. Build RED and BLUE subgraphs as IRDME Graph objects
  4. Build a DimensionalSpace over vertex profiles
  5. Run IRDME dimensional analysis:
       - suspicious_connections(red_graph): red edges whose endpoints are
         dimensionally far apart -> candidates to flip blue
       - suspicious_connections(blue_graph): blue edges whose endpoints are
         dimensionally far apart -> candidates to flip red
       - dimension_projection_clusters: vertex type classification
       - hyperplane_split: does hub-balance predict coloring?
  6. Score flip candidates by: dim_distance * violation_count
  7. Run IRDME-guided search: always flip the highest-scored candidate

Usage:
  python scripts/ramsey_irdme.py --n 42 --s 5 --iter 2000 --restarts 10
  python scripts/ramsey_irdme.py --n 42 --s 5 --analyze-only
  python scripts/ramsey_irdme.py --n 42 --s 5 --load datasets/RAMSEY_K42_solution.json
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

# IRDME engine
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model import Graph, Item, Relation
from core.space import DimensionalSpace, Axis
from core.algorithms.dimensional import (
    suspicious_connections, dimension_projection_clusters, hyperplane_split
)


# ---------------------------------------------------------------------------
# Ramsey clique structure (reused from ramsey_local.py logic)
# ---------------------------------------------------------------------------

def build_structure(n: int, s: int):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_idx = {e: k for k, e in enumerate(edges)}
    clique_size_edges = s * (s - 1) // 2
    cliques = []
    e2c = [[] for _ in range(len(edges))]
    t0 = time.time()
    print(f"  Building clique structure: n={n}, s={s}")
    for ci, cv in enumerate(combinations(range(n), s)):
        ce = []
        for a in range(s):
            for b in range(a + 1, s):
                ek = edge_idx[(cv[a], cv[b])]
                ce.append(ek)
                e2c[ek].append(ci)
        cliques.append(ce)
        if ci % 100_000 == 0 and ci > 0:
            print(f"    {ci:,} cliques ({time.time()-t0:.1f}s)")
    print(f"  Total cliques: {len(cliques):,}  ({time.time()-t0:.1f}s)")
    return edges, edge_idx, cliques, e2c


def init_counts(assignment, cliques, clique_size_edges):
    clique_red = [0] * len(cliques)
    clique_blue = [0] * len(cliques)
    violation_set = set()
    for ci, ce in enumerate(cliques):
        for ek in ce:
            if assignment[ek] == 1:
                clique_red[ci] += 1
            else:
                clique_blue[ci] += 1
        if clique_red[ci] == clique_size_edges or clique_blue[ci] == clique_size_edges:
            violation_set.add(ci)
    return clique_red, clique_blue, violation_set


# ---------------------------------------------------------------------------
# Seed generation (Paley + FPL hub balance)
# ---------------------------------------------------------------------------

def _paley_red(p: int) -> set[tuple[int, int]]:
    qr = {(x * x) % p for x in range(1, p)}
    return {(i, j) for i in range(p) for j in range(i + 1, p) if (j - i) % p in qr}


def structured_seed(n, s, rng, edges, edge_idx, cliques, e2c):
    cse = s * (s - 1) // 2
    assignment = [-1] * len(edges)
    clique_red = [0] * len(cliques)
    clique_blue = [0] * len(cliques)
    red_deg = [0] * n
    blue_deg = [0] * n
    target = (n - 1) / 2.0

    def _assign(ek, color):
        assignment[ek] = color
        i, j = edges[ek]
        if color == 1:
            red_deg[i] += 1; red_deg[j] += 1
            for ci in e2c[ek]: clique_red[ci] += 1
        else:
            blue_deg[i] += 1; blue_deg[j] += 1
            for ci in e2c[ek]: clique_blue[ci] += 1

    # Paley(17) base
    pr = _paley_red(17)
    for i in range(17):
        for j in range(i + 1, 17):
            _assign(edge_idx[(i, j)], 1 if (i, j) in pr else 0)

    # Shuffled greedy with hub-balance tiebreak for remaining edges
    remaining = [ek for ek in range(len(edges)) if assignment[ek] == -1]
    rng.shuffle(remaining)
    for ek in remaining:
        i, j = edges[ek]
        best, best_score = 0, float('inf')
        for color in [0, 1]:
            nv = sum(
                1 for ci in e2c[ek]
                if (clique_red[ci] + (1 if color == 1 else 0)) == cse
                or (clique_blue[ci] + (1 if color == 0 else 0)) == cse
            )
            imbal = (
                abs(red_deg[i] + 1 - target) + abs(red_deg[j] + 1 - target)
                if color == 1 else
                abs(blue_deg[i] + 1 - target) + abs(blue_deg[j] + 1 - target)
            )
            score = nv + 0.001 * imbal
            if score < best_score:
                best_score = score; best = color
        _assign(ek, best)
    return assignment


# ---------------------------------------------------------------------------
# Vertex / edge dimension computation
# ---------------------------------------------------------------------------

def compute_dimensions(n, s, assignment, edges, edge_idx, cliques, e2c,
                       clique_red, clique_blue, violation_set):
    """
    Compute per-vertex and per-edge dimensions for IRDME analysis.

    Vertex dimensions:
      red_degree       -- number of red edges
      blue_degree      -- number of blue edges
      red_triangles    -- triangles in red subgraph through this vertex
      blue_triangles   -- triangles in blue subgraph through this vertex
      red_violations   -- violated (all-red) cliques containing this vertex
      blue_violations  -- violated (all-blue) cliques containing this vertex
      hub_imbalance    -- |red_degree - blue_degree|

    Edge dimensions:
      color            -- 1=red, 0=blue
      violation_count  -- number of violated cliques containing this edge
      shared_red       -- shared red neighbors between endpoints
      shared_blue      -- shared blue neighbors between endpoints
    """
    cse = s * (s - 1) // 2

    # Adjacency by color per vertex
    red_nbrs = [set() for _ in range(n)]
    blue_nbrs = [set() for _ in range(n)]
    for ek, (i, j) in enumerate(edges):
        if assignment[ek] == 1:
            red_nbrs[i].add(j); red_nbrs[j].add(i)
        else:
            blue_nbrs[i].add(j); blue_nbrs[j].add(i)

    # Vertex violation counts
    v_red_viol = [0] * n
    v_blue_viol = [0] * n
    for ci in violation_set:
        is_red_viol = clique_red[ci] == cse
        cv = set()
        for ek in cliques[ci]:
            cv.add(edges[ek][0]); cv.add(edges[ek][1])
        for v in cv:
            if is_red_viol:
                v_red_viol[v] += 1
            else:
                v_blue_viol[v] += 1

    # Triangle counts per vertex
    red_tri = [0] * n
    blue_tri = [0] * n
    for v in range(n):
        rn = list(red_nbrs[v])
        for a in range(len(rn)):
            for b in range(a + 1, len(rn)):
                if rn[b] in red_nbrs[rn[a]]:
                    red_tri[v] += 1
        bn = list(blue_nbrs[v])
        for a in range(len(bn)):
            for b in range(a + 1, len(bn)):
                if bn[b] in blue_nbrs[bn[a]]:
                    blue_tri[v] += 1

    # Edge violation counts
    e_viol = [0] * len(edges)
    for ci in violation_set:
        for ek in cliques[ci]:
            e_viol[ek] += 1

    # Assemble vertex dims
    v_dims = {}
    for v in range(n):
        rd = len(red_nbrs[v])
        bd = len(blue_nbrs[v])
        v_dims[v] = {
            "red_degree":      float(rd),
            "blue_degree":     float(bd),
            "red_triangles":   float(red_tri[v]),
            "blue_triangles":  float(blue_tri[v]),
            "red_violations":  float(v_red_viol[v]),
            "blue_violations": float(v_blue_viol[v]),
            "hub_imbalance":   float(abs(rd - bd)),
        }

    # Assemble edge dims
    e_dims = {}
    for ek, (i, j) in enumerate(edges):
        shared_r = len(red_nbrs[i] & red_nbrs[j])
        shared_b = len(blue_nbrs[i] & blue_nbrs[j])
        e_dims[ek] = {
            "color":         float(assignment[ek]),
            "violation_count": float(e_viol[ek]),
            "shared_red":    float(shared_r),
            "shared_blue":   float(shared_b),
        }

    return v_dims, e_dims, red_nbrs, blue_nbrs


# ---------------------------------------------------------------------------
# Build IRDME Graph objects
# ---------------------------------------------------------------------------

def build_irdme_graphs(n, edges, assignment, v_dims, e_dims):
    """
    Build three IRDME Graph objects:
      - red_graph:  only the red edges (for suspicious_connections analysis)
      - blue_graph: only the blue edges
      - full_graph: all edges typed red/blue (for clustering)
    All graphs share the same vertex items with their full dimension vector.
    """
    items = {
        str(v): Item(
            id=str(v),
            type="vertex",
            label=f"v{v}",
            dimensions=v_dims[v],
        )
        for v in range(n)
    }

    def _make_graph(color_filter):
        g = Graph()
        for iid, item in items.items():
            g.add_item(item)
        for ek, (i, j) in enumerate(edges):
            if color_filter is None or assignment[ek] == color_filter:
                color = "red" if assignment[ek] == 1 else "blue"
                g.add_relation(Relation(
                    id=f"e{ek}",
                    from_id=str(i),
                    to_id=str(j),
                    type=color,
                    directed=False,
                    dimensions=e_dims[ek],
                ))
        return g

    return _make_graph(1), _make_graph(0), _make_graph(None)


# ---------------------------------------------------------------------------
# DimensionalSpace definition for Ramsey vertex analysis
# ---------------------------------------------------------------------------

def build_space(n: int) -> DimensionalSpace:
    """
    4-axis vertex space (the D1-D4 of IRDME for Ramsey):
      D1: red_degree       -- hub balance axis (target: ~(n-1)/2)
      D2: blue_degree      -- complementary hub axis
      D3: red_violations   -- structural health in red layer
      D4: blue_violations  -- structural health in blue layer

    Violations axes get higher weight because they directly measure
    how far we are from a valid coloring.
    """
    half = (n - 1) / 2
    return DimensionalSpace(
        name="ramsey_vertex_space",
        description=(
            f"4D structural space for Ramsey K_{n} vertices. "
            "D1/D2: hub balance. D3/D4: violation load. "
            "Valid colorings cluster at D1???D2???{half:.0f}, D3=D4=0."
        ),
        axes=[
            Axis("red_degree",      label="Red degree (D1)",
                 range_min=0, range_max=n - 1, weight=1.5),
            Axis("blue_degree",     label="Blue degree (D2)",
                 range_min=0, range_max=n - 1, weight=1.5),
            Axis("red_violations",  label="Red violation load (D3)",
                 range_min=0, range_max=max(1, n * 10), weight=3.0),
            Axis("blue_violations", label="Blue violation load (D4)",
                 range_min=0, range_max=max(1, n * 10), weight=3.0),
        ],
        distance_metric="weighted_euclidean",
    )


# ---------------------------------------------------------------------------
# IRDME flip-candidate scoring
# ---------------------------------------------------------------------------

def irdme_flip_candidates(
    red_graph: Graph, blue_graph: Graph, space: DimensionalSpace,
    edges: list, edge_idx: dict, e_dims: dict, top_k: int = 50
) -> list[tuple[int, float, str]]:
    """
    Use IRDME suspicious_connections to rank flip candidates.

    For the red subgraph: suspicious red edges (endpoints dimensionally far)
    -> candidates to flip blue.
    For the blue subgraph: suspicious blue edges (endpoints dimensionally far)
    -> candidates to flip red.

    Each candidate is scored by: dim_distance * (1 + violation_count).
    Returns list of (edge_idx, score, direction) sorted by score descending.
    """
    candidates = []

    num_edges = len(edges)

    # Suspicious red edges (flip to blue)
    sc_red = suspicious_connections(red_graph, space, top_n=num_edges)
    for entry in sc_red.get("top_suspicious", []):
        i, j = int(entry["item_a"]), int(entry["item_b"])
        key = (min(i, j), max(i, j))
        if key in edge_idx:
            ek = edge_idx[key]
            dim_d = entry["dim_distance"]
            viol = e_dims[ek]["violation_count"]
            score = dim_d * (1 + viol)
            candidates.append((ek, score, "red->blue"))

    # Suspicious blue edges (flip to red)
    sc_blue = suspicious_connections(blue_graph, space, top_n=num_edges)
    for entry in sc_blue.get("top_suspicious", []):
        i, j = int(entry["item_a"]), int(entry["item_b"])
        key = (min(i, j), max(i, j))
        if key in edge_idx:
            ek = edge_idx[key]
            dim_d = entry["dim_distance"]
            viol = e_dims[ek]["violation_count"]
            score = dim_d * (1 + viol)
            candidates.append((ek, score, "blue->red"))

    # Keep only one entry per edge (higher score wins)
    best: dict[int, tuple[int, float, str]] = {}
    for ek, score, direction in candidates:
        if ek not in best or score > best[ek][1]:
            best[ek] = (ek, score, direction)

    return sorted(best.values(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Print analysis summary
# ---------------------------------------------------------------------------

def print_analysis(n, s, violation_set, v_dims, space,
                   red_graph, blue_graph, full_graph, flip_candidates):
    print(f"\n{'='*68}")
    print(f"  IRDME DIMENSIONAL ANALYSIS -- K_{n}, s={s}")
    print(f"  Total violations: {len(violation_set)}")
    print(f"{'='*68}")

    # Vertex profile summary
    all_rd = [v_dims[v]["red_degree"] for v in range(n)]
    all_bd = [v_dims[v]["blue_degree"] for v in range(n)]
    all_rv = [v_dims[v]["red_violations"] for v in range(n)]
    all_bv = [v_dims[v]["blue_violations"] for v in range(n)]
    print(f"\n  Vertex profile (n={n}):")
    print(f"    red_degree:     min={min(all_rd):.0f}  max={max(all_rd):.0f}  "
          f"mean={sum(all_rd)/n:.1f}  target={(n-1)/2:.1f}")
    print(f"    blue_degree:    min={min(all_bd):.0f}  max={max(all_bd):.0f}  "
          f"mean={sum(all_bd)/n:.1f}")
    print(f"    red_violations: min={min(all_rv):.0f}  max={max(all_rv):.0f}  "
          f"mean={sum(all_rv)/n:.1f}")
    print(f"    blue_violations:min={min(all_bv):.0f}  max={max(all_bv):.0f}  "
          f"mean={sum(all_bv)/n:.1f}")

    # Hub balance distribution
    imbalanced = sum(1 for v in range(n) if abs(all_rd[v] - all_bd[v]) > 5)
    print(f"    hub_imbalance>5: {imbalanced}/{n} vertices "
          f"({'healthy' if imbalanced < n//4 else 'UNBALANCED'})")

    # Clusters
    clusters = dimension_projection_clusters(full_graph, space, k=3)
    print(f"\n  Vertex clusters (k=3 in D1-D4 space):")
    for cl in clusters.get("clusters", []):
        ctr = cl["centroid"]
        print(f"    Cluster {cl['cluster_id']}: {cl['size']} vertices  "
              f"rd={ctr.get('red_degree',0):.1f}  "
              f"bd={ctr.get('blue_degree',0):.1f}  "
              f"rv={ctr.get('red_violations',0):.1f}  "
              f"bv={ctr.get('blue_violations',0):.1f}")

    # Hyperplane split
    hs = hyperplane_split(red_graph, space, "red_degree", threshold=(n - 1) / 2)
    print(f"\n  Hyperplane split (red_degree >= {(n-1)/2:.1f}):")
    print(f"    cross_ratio = {hs.get('cross_ratio', '?')}  "
          f"-> {hs.get('interpretation', '')[:80]}")

    # Top flip candidates
    print(f"\n  Top IRDME flip candidates (dim_distance * violation_count):")
    for ek, score, direction in flip_candidates[:15]:
        print(f"    edge {ek:>4}  score={score:6.2f}  {direction}")

    print()


# ---------------------------------------------------------------------------
# IRDME-guided local search
# ---------------------------------------------------------------------------

def _flip_edge(ek, assignment, clique_red, clique_blue, violation_set,
               e2c, cliques, cse):
    is_red = assignment[ek] == 1
    assignment[ek] = 0 if is_red else 1
    for ci in e2c[ek]:
        old_rc, old_bc = clique_red[ci], clique_blue[ci]
        if is_red:
            clique_red[ci] -= 1; clique_blue[ci] += 1
        else:
            clique_red[ci] += 1; clique_blue[ci] -= 1
        was_v = old_rc == cse or old_bc == cse
        now_v = clique_red[ci] == cse or clique_blue[ci] == cse
        if was_v and not now_v:
            violation_set.discard(ci)
        elif not was_v and now_v:
            violation_set.add(ci)


def _delta(ek, assignment, clique_red, clique_blue, violation_set, e2c, cliques, cse):
    is_red = assignment[ek] == 1
    fixed = new = 0
    for ci in e2c[ek]:
        rc, bc = clique_red[ci], clique_blue[ci]
        rc2, bc2 = (rc - 1, bc + 1) if is_red else (rc + 1, bc - 1)
        if (rc == cse or bc == cse) and not (rc2 == cse or bc2 == cse):
            fixed += 1
        elif not (rc == cse or bc == cse) and (rc2 == cse or bc2 == cse):
            new += 1
    return fixed - new


def irdme_guided_search(
    n, s, edges, edge_idx, cliques, e2c, assignment,
    space, max_iter=2000, reanalyze_every=200, rng=None, verbose=True
):
    """
    IRDME-guided local search.

    Every `reanalyze_every` iterations, re-run dimensional analysis to get
    fresh flip candidates. Between re-analyses, work through the candidate list
    from top to bottom, taking only moves that reduce or maintain violations
    (or accept small uphill moves when stuck).

    This replaces the random violated-clique selection of standard tabu search
    with dimensionally-informed edge selection: we flip edges that are
    structurally wrong for their dimensional position first.
    """
    if rng is None:
        rng = random.Random()

    cse = s * (s - 1) // 2
    clique_red, clique_blue, violation_set = init_counts(assignment, cliques, cse)

    best_assignment = assignment[:]
    best_violations = len(violation_set)
    flip_candidates = []
    candidate_ptr = 0
    last_reanalyze = -reanalyze_every  # trigger immediately

    print(f"\n  IRDME-guided search: {max_iter} iterations, reanalyze every {reanalyze_every}")
    print(f"  Starting violations: {len(violation_set)}")

    t0 = time.time()

    for it in range(max_iter):
        v_count = len(violation_set)

        if v_count < best_violations:
            best_violations = v_count
            best_assignment = assignment[:]

        if v_count == 0:
            print(f"  [iter {it}] SOLVED! Valid coloring found.")
            return assignment, it, 0, assignment

        if verbose and it % 100 == 0:
            print(f"  iter {it:>6}  violations={v_count:>5}  best={best_violations:>5}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # Re-run dimensional analysis periodically
        if it - last_reanalyze >= reanalyze_every:
            v_dims, e_dims, _, _ = compute_dimensions(
                n, s, assignment, edges, edge_idx, cliques, e2c,
                clique_red, clique_blue, violation_set
            )
            red_g, blue_g, full_g = build_irdme_graphs(n, edges, assignment, v_dims, e_dims)
            flip_candidates = irdme_flip_candidates(
                red_g, blue_g, space, edges, edge_idx, e_dims, top_k=100
            )
            candidate_ptr = 0
            last_reanalyze = it
            if verbose:
                print(f"  [iter {it}] IRDME reanalysis -> {len(flip_candidates)} candidates")

        # Try the next IRDME-recommended flip
        flipped = False
        attempts = 0
        while candidate_ptr < len(flip_candidates) and attempts < 10:
            ek, score, direction = flip_candidates[candidate_ptr]
            candidate_ptr += 1
            attempts += 1

            # Skip if this edge's current color doesn't match the recommendation
            current_red = assignment[ek] == 1
            if direction == "red->blue" and not current_red:
                continue
            if direction == "blue->red" and current_red:
                continue

            d = _delta(ek, assignment, clique_red, clique_blue, violation_set, e2c, cliques, cse)
            if d >= 0:  # improving or neutral
                _flip_edge(ek, assignment, clique_red, clique_blue, violation_set, e2c, cliques, cse)
                flipped = True
                break

        if not flipped:
            # No good IRDME candidate: fall back to random violated-clique flip
            if violation_set:
                ci = rng.choice(list(violation_set))
                ce = cliques[ci]
                ek = max(ce, key=lambda e: _delta(e, assignment, clique_red, clique_blue,
                                                   violation_set, e2c, cliques, cse))
                _flip_edge(ek, assignment, clique_red, clique_blue, violation_set, e2c, cliques, cse)
            candidate_ptr = len(flip_candidates)  # force reanalysis on next round

    return None, max_iter, best_violations, best_assignment


# ---------------------------------------------------------------------------
# Save dataset
# ---------------------------------------------------------------------------

def save_dataset(n, s, assignment, edges, v_dims, e_dims, path):
    items = [
        {"id": str(v), "type": "vertex", "label": f"v{v}", "dimensions": v_dims[v]}
        for v in range(n)
    ]
    relations = [
        {
            "from": str(i), "to": str(j),
            "type": "red" if assignment[ek] == 1 else "blue",
            "directed": False,
            "dimensions": e_dims[ek],
        }
        for ek, (i, j) in enumerate(edges)
    ]
    data = {
        "meta": {
            "name": f"Ramsey K_{n} IRDME coloring (avoid K_{s})",
            "domain": "mathematics", "subdomain": "ramsey_theory",
            "n": n, "s": s,
            "generated_by": "ramsey_irdme.py",
        },
        "items": items,
        "relations": relations,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, n: int, s: int, assignment: list[int],
                    violations: int, restart: int, total_iters: int,
                    snap_counter: int | None = None) -> None:
    data = {
        "n": n, "s": s,
        "violations": violations,
        "restart": restart,
        "total_iters": total_iters,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "assignment": assignment,
    }
    # Rolling checkpoint — overwrites (for resume)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    Path(tmp).replace(path)
    print(f"  [checkpoint] saved -> {path}  "
          f"(violations={violations}, restart={restart}, iters={total_iters})")

    # Immutable snapshot — new file per new best, never overwritten
    if snap_counter is not None:
        stem = Path(path).stem.replace("_checkpoint", "")
        snap_name = f"{stem}_snap_{snap_counter:03d}_v{violations}.json"
        snap_path = Path(path).parent / snap_name
        with open(snap_path, "w") as f:
            json.dump(data, f)
        print(f"  [snapshot]   saved -> {snap_path}")


def load_checkpoint(path: str) -> dict | None:
    if not Path(path).exists():
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"  [checkpoint] loaded <- {path}")
    print(f"    violations={data['violations']}  restart={data['restart']}  "
          f"iters={data['total_iters']}  saved={data.get('timestamp','?')}")
    return data


def latest_snap_counter(ckpt_path: str) -> int:
    """Scan existing snapshot files and return the next counter value."""
    stem = Path(ckpt_path).stem.replace("_checkpoint", "")
    parent = Path(ckpt_path).parent
    existing = list(parent.glob(f"{stem}_snap_*.json"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        parts = p.stem.split("_snap_")
        if len(parts) == 2:
            try:
                nums.append(int(parts[1].split("_")[0]))
            except ValueError:
                pass
    return max(nums) + 1 if nums else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IRDME-guided Ramsey coloring search")
    parser.add_argument("--n", type=int, default=42)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--iter", type=int, default=2000, help="Iterations per restart")
    parser.add_argument("--restarts", type=int, default=500,
                        help="Total restarts (default 500 for long runs)")
    parser.add_argument("--reanalyze", type=int, default=200,
                        help="Re-run IRDME analysis every N iterations")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Analyze the seed coloring without running search")
    parser.add_argument("--load", default=None,
                        help="Load a saved coloring JSON instead of generating a seed")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint file path. If exists: resume from it. "
                             "Updated after every new best and every restart.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    n, s = args.n, args.s
    rng = random.Random(args.seed)

    # Default checkpoint path
    ckpt_path = args.checkpoint or f"datasets/RAMSEY_K{n}_S{s}_irdme_checkpoint.json"
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  IRDME Ramsey Search -- K_{n}, avoid monochromatic K_{s}")
    print(f"  Restarts: {args.restarts}   Iters/restart: {args.iter}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*68}\n")

    # Build clique structure (always needed, ~2.5s)
    edges, edge_idx, cliques, e2c = build_structure(n, s)
    cse = s * (s - 1) // 2
    space = build_space(n)

    # --- Resume from checkpoint if available ---
    ckpt = load_checkpoint(ckpt_path)
    overall_best = float('inf')
    overall_best_assignment = None
    start_restart = 0
    total_iters = 0

    if ckpt and ckpt["n"] == n and ckpt["s"] == s:
        overall_best = ckpt["violations"]
        overall_best_assignment = ckpt["assignment"]
        start_restart = ckpt["restart"] + 1  # resume AFTER the saved restart
        total_iters = ckpt["total_iters"]
        print(f"  Resuming from restart {start_restart+1}/{args.restarts}, "
              f"best so far: {overall_best} violations\n")
    elif ckpt:
        print(f"  Checkpoint is for K_{ckpt['n']} s={ckpt['s']}, ignoring (wrong graph).\n")
        ckpt = None

    # Snapshot counter: pick up where existing snapshots left off
    snap_counter = latest_snap_counter(ckpt_path)

    # Track restarts since last breakthrough for exploit/explore scheduling
    restart_of_last_best = start_restart  # reset to now on resume

    t_start = time.time()

    try:
        for restart in range(start_restart, args.restarts):
            print(f"\n{'-'*50}")
            print(f"  Restart {restart+1}/{args.restarts}  "
                  f"(elapsed {time.time()-t_start:.0f}s, total iters={total_iters})")

            num_edges = len(edges)
            restarts_since_best = restart - restart_of_last_best

            # --- Seed strategy ---
            # First restart after resume: use checkpoint best directly
            if restart == start_restart and overall_best_assignment is not None:
                assignment = overall_best_assignment[:]
                cr, cb, vs = init_counts(assignment, cliques, cse)
                print(f"  Seed: checkpoint best ({len(vs)} violations)")

            elif args.load and restart == 0:
                with open(args.load) as f:
                    raw = json.load(f)
                color_map = {r["from"] + "-" + r["to"]: r["type"] for r in raw["relations"]}
                assignment = []
                for i, j in edges:
                    key = f"{i}-{j}"
                    t = color_map.get(key) or color_map.get(f"{j}-{i}", "blue")
                    assignment.append(1 if t == "red" else 0)
                print(f"  Seed: loaded from {args.load}")

            elif overall_best_assignment is not None and restarts_since_best < 6:
                # Breakthrough recently: exploit hard.
                # Run 6 perturbation restarts before allowing any fresh seed.
                # Perturbation size escalates: 5%, 10%, 15%, 5%, 10%, 15%
                frac = [0.05, 0.10, 0.15][restarts_since_best % 3]
                n_flip = max(5, int(num_edges * frac))
                assignment = overall_best_assignment[:]
                for ek in rng.sample(range(num_edges), n_flip):
                    assignment[ek] ^= 1
                cr, cb, vs = init_counts(assignment, cliques, cse)
                print(f"  Seed: best+perturb ({frac*100:.0f}%, {n_flip} flips, "
                      f"{restarts_since_best+1}/6 exploit  [{len(vs)} violations])")

            elif overall_best_assignment is not None and restarts_since_best % 4 != 0:
                # Stagnating: mostly keep perturbing, but allow one fresh seed
                # every 4th restart to avoid getting permanently stuck.
                frac = [0.10, 0.20, 0.30][min(2, restarts_since_best // 10)]
                n_flip = max(5, int(num_edges * frac))
                assignment = overall_best_assignment[:]
                for ek in rng.sample(range(num_edges), n_flip):
                    assignment[ek] ^= 1
                cr, cb, vs = init_counts(assignment, cliques, cse)
                print(f"  Seed: best+perturb ({frac*100:.0f}%, stagnated {restarts_since_best} "
                      f"restarts  [{len(vs)} violations])")

            else:
                # Fresh Paley+FPL seed (1 in 4 restarts when stagnating, or no best yet)
                assignment = structured_seed(n, s, rng, edges, edge_idx, cliques, e2c)
                cr, cb, vs = init_counts(assignment, cliques, cse)
                print(f"  Seed: Paley+FPL fresh [{len(vs)} violations]")

            # --- Dimensional analysis ---
            v_dims, e_dims, _, _ = compute_dimensions(
                n, s, assignment, edges, edge_idx, cliques, e2c, cr, cb, vs
            )
            red_g, blue_g, full_g = build_irdme_graphs(n, edges, assignment, v_dims, e_dims)
            flip_candidates = irdme_flip_candidates(
                red_g, blue_g, space, edges, edge_idx, e_dims
            )

            if restart == start_restart or args.analyze_only:
                print_analysis(n, s, vs, v_dims, space, red_g, blue_g, full_g, flip_candidates)

            if args.analyze_only:
                out = args.out or f"datasets/RAMSEY_K{n}_S{s}_irdme.json"
                save_dataset(n, s, assignment, edges, v_dims, e_dims, out)
                return

            # --- IRDME-guided search ---
            result, iters_done, best_v, best_coloring = irdme_guided_search(
                n=n, s=s, edges=edges, edge_idx=edge_idx,
                cliques=cliques, e2c=e2c, assignment=assignment,
                space=space, max_iter=args.iter, reanalyze_every=args.reanalyze,
                rng=rng, verbose=True,
            )
            total_iters += iters_done

            if best_v < overall_best:
                overall_best = best_v
                overall_best_assignment = best_coloring  # always the actual best, not end state
                restart_of_last_best = restart  # lock exploit mode for next 6 restarts
                print(f"\n  *** New global best: {overall_best} violations ***")
                save_checkpoint(ckpt_path, n, s, overall_best_assignment,
                                overall_best, restart, total_iters,
                                snap_counter=snap_counter)
                snap_counter += 1

            # Save checkpoint after every restart (captures progress even without new best)
            elif restart % 5 == 0:
                save_checkpoint(ckpt_path, n, s,
                                overall_best_assignment or best_coloring,
                                overall_best, restart, total_iters)

            if result is not None:
                print(f"\n{'='*60}")
                print(f"  SOLVED: valid K_{n} coloring found! R({s},{s}) > {n}")
                print(f"  Total time: {time.time()-t_start:.1f}s, iters: {total_iters}")
                out = args.out or f"datasets/RAMSEY_K{n}_S{s}_irdme_solution.json"
                cr2, cb2, vs2 = init_counts(result, cliques, cse)
                vd2, ed2, _, _ = compute_dimensions(
                    n, s, result, edges, edge_idx, cliques, e2c, cr2, cb2, vs2
                )
                save_dataset(n, s, result, edges, vd2, ed2, out)
                Path(ckpt_path).unlink(missing_ok=True)
                return

    except KeyboardInterrupt:
        print(f"\n\n  Interrupted by user. Saving checkpoint...")
        if overall_best_assignment is not None:
            save_checkpoint(ckpt_path, n, s, overall_best_assignment,
                            overall_best, restart, total_iters)
            print(f"  Resume with:  python scripts/ramsey_irdme.py "
                  f"--n {n} --s {s} --iter {args.iter} --restarts {args.restarts} "
                  f"--checkpoint {ckpt_path}")
        return

    print(f"\n{'='*68}")
    print(f"  Search complete. Best violations: {overall_best}")
    print(f"  Total time: {time.time()-t_start:.1f}s, total iters: {total_iters}")

    if overall_best_assignment is not None:
        cr, cb, vs = init_counts(overall_best_assignment, cliques, cse)
        vd, ed, _, _ = compute_dimensions(
            n, s, overall_best_assignment, edges, edge_idx, cliques, e2c, cr, cb, vs
        )
        out = args.out or f"datasets/RAMSEY_K{n}_S{s}_irdme.json"
        save_dataset(n, s, overall_best_assignment, edges, vd, ed, out)


if __name__ == "__main__":
    main()
