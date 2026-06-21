"""
ramsey_bipartite_to_irdme.py  --  Bipartite IRDME: vertices + orbit types as items.

Design: "combination of both" -- vertices AND orbit types are items.

Items (83 total for K_42 bi-circulant):
  v0..v41   -- 42 vertex items  (type: "vertex")
  T0..T40   -- 41 orbit-type items (type: "orbit_type")

Shared dimensions (same name for both, 0 where not applicable):
  violation_exposure   -- vertex: # K5s containing it; orbit: # active K5 subgraphs
  structural_weight    -- vertex: total degree (n-1=41); orbit: # canonical K5 orbits
  color_balance        -- vertex: |red_deg - blue_deg| (0=perfectly balanced);
                          orbit: 2*color - 1 (red=+1, blue=-1)
  is_orbit             -- 0 for vertices, 1 for orbit types (for atlas filtering)

Layers:
  vertex_violation_layer   -- vertex <-> vertex: co-appear in >=1 K5 violation
  orbit_conflict_layer     -- orbit <-> orbit: co-appear in >=1 K5 subgraph
  vertex_orbit_layer       -- vertex <-> orbit: vertex appears in a K5 violation
                              that contains an edge of this orbit type

FPL questions:
  1. violation_exposure across layers: do vertex violation hubs bind to orbit violation hubs?
     (Does v42 = vertex #1 bind disproportionately to C12-C16 = orbit #1?)
  2. Atlas: which vertices are structurally coupled to which orbit type classes?
     (Are cross-half C types the orbit-partners of universal hub vertices?)

Usage:
  python scripts/ramsey_bipartite_to_irdme.py datasets/RAMSEY_K42_S5_bicirculant_84v.json
  python irdme.py law   examples/bipartite_RAMSEY_K42_S5_bicirculant_84v.json
  python irdme.py atlas examples/bipartite_RAMSEY_K42_S5_bicirculant_84v.json
"""

import json, itertools, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------------------
# Z_21 orbit utilities
# ---------------------------------------------------------------------------

N       = 42
N1      = 21
N_TYPES = 41

def vertex_decompose(v):
    return v % N1, v // N1

def edge_orbit(u, v):
    if u > v:
        u, v = v, u
    pu, hu = vertex_decompose(u)
    pv, hv = vertex_decompose(v)
    if hu == hv:
        diff = (pv - pu) % N1
        if diff > N1 // 2:
            diff = N1 - diff
        k = diff - 1
        return k if hu == 0 else k + 10
    else:
        if hu == 1:
            pu, pv = pv, pu
        diff = (pv - pu) % N1
        return 20 + diff

def orbit_label(t):
    if t < 10:  return f"A{t+1}"
    elif t < 20: return f"B{t-9}"
    else:        return f"C{t-20}"

def orbit_class(t):
    if t < 10:  return "within_half0"
    elif t < 20: return "within_half1"
    else:        return "cross_half"

# ---------------------------------------------------------------------------
# Full K5 enumeration (actual subgraphs, not orbits)
# ---------------------------------------------------------------------------

def find_all_k5(n, assignment_flat):
    """Find all monochromatic K5 subgraphs. Returns list of (color, [v0..v4])."""
    adj = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            adj[(i,j)] = assignment_flat[idx]
            idx += 1
    violations = []
    for combo in itertools.combinations(range(n), 5):
        colors = [adj[(combo[i], combo[j])] for i in range(5) for j in range(i+1, 5)]
        if all(c == 1 for c in colors):
            violations.append((1, list(combo)))
        elif all(c == 0 for c in colors):
            violations.append((0, list(combo)))
    return violations

# ---------------------------------------------------------------------------
# Build canonical K5 orbits for orbit-type analysis
# ---------------------------------------------------------------------------

def build_k5_orbits():
    print("  Building canonical K5 orbits...", end="", flush=True)
    t0 = time.time()
    seen = {}
    for combo in itertools.combinations(range(N), 5):
        min_rep = combo
        for k in range(1, N1):
            rotated = tuple(sorted(((v % N1 + k) % N1 + (v // N1) * N1) for v in combo))
            if rotated < min_rep:
                min_rep = rotated
        if min_rep not in seen:
            mask = 0
            for i in range(5):
                for j in range(i+1, 5):
                    mask |= (1 << edge_orbit(min_rep[i], min_rep[j]))
            seen[min_rep] = mask
    print(f" {len(seen)} orbits  ({time.time()-t0:.1f}s)")
    return list(seen.items())

# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_path, output_path=None):
    data = json.load(open(input_path))
    n = data["n"]
    declared_v = data["violations"]
    assignment_flat = data["assignment"]

    if n != N:
        print(f"  Note: designed for K_42 bi-circulant (n=42), got n={n}")

    print(f"  Loaded: n={n}, declared_violations={declared_v}")

    # Recover orbit type assignments
    types = [0] * N_TYPES
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            types[edge_orbit(i, j)] = assignment_flat[idx]
            idx += 1
    assignment_int = sum(int(types[t]) << t for t in range(N_TYPES))

    # ---- Canonical K5 orbits ----
    canon_list = build_k5_orbits()
    n_orbits = len(canon_list)
    masks_np = np.array([m for _, m in canon_list], dtype=np.int64)

    def is_mono(mask, asgn_int):
        bits = asgn_int & mask
        return bits == 0 or bits == mask

    # ---- Actual K5 violations ----
    print(f"  Finding actual K5 violations...", end="", flush=True)
    t0 = time.time()
    violations = find_all_k5(n, assignment_flat)
    print(f" {len(violations)}  ({time.time()-t0:.1f}s)")

    # ---- Per-vertex stats ----
    vcount = [0] * n
    red_deg  = [0] * n
    blue_deg = [0] * n

    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = assignment_flat[idx]
            if c == 1:
                red_deg[i]  += 1
                red_deg[j]  += 1
            else:
                blue_deg[i] += 1
                blue_deg[j] += 1
            idx += 1

    for color, verts in violations:
        for v in verts:
            vcount[v] += 1

    # ---- Per-orbit stats ----
    type_to_canons = [[] for _ in range(N_TYPES)]
    for ci, (_, mask) in enumerate(canon_list):
        for t in range(N_TYPES):
            if mask >> t & 1:
                type_to_canons[t].append(ci)

    def count_violations_exact(asgn_int):
        return sum(N1 for mask in masks_np if is_mono(int(mask), asgn_int))

    base_v = count_violations_exact(assignment_int)

    orbit_data = []
    for t in range(N_TYPES):
        canons = type_to_canons[t]
        k5_mono = sum(N1 for ci in canons if is_mono(int(masks_np[ci]), assignment_int))
        delta_v = count_violations_exact(assignment_int ^ (1 << t)) - base_v
        orbit_data.append({
            "color":             types[t],
            "k5_mono_exposure":  k5_mono,
            "n_canon_orbits":    len(canons),
            "delta_if_flip":     delta_v,
        })

    # ---- Build vertex-orbit participation from actual violations ----
    # For each actual K5 violation, each of its 5 vertices is connected to
    # each of its 10 edge types (orbit types)
    vertex_orbit_weight = defaultdict(int)  # (v, t) -> count
    for color, verts in violations:
        involved_types = set()
        for i in range(5):
            for j in range(i+1, 5):
                involved_types.add(edge_orbit(verts[i], verts[j]))
        for v in verts:
            for t in involved_types:
                vertex_orbit_weight[(v, t)] += 1

    # ---- Vertex co-violation graph ----
    vertex_co_viol = defaultdict(int)
    for color, verts in violations:
        for u, v in itertools.combinations(sorted(verts), 2):
            vertex_co_viol[(u, v)] += 1

    # ---- Orbit co-conflict graph ----
    orbit_conflict = defaultdict(int)
    orbit_mono_conflict = defaultdict(int)
    for ci, (_, mask) in enumerate(canon_list):
        involved = [t for t in range(N_TYPES) if mask >> t & 1]
        mono = is_mono(int(mask), assignment_int)
        for a, b in itertools.combinations(involved, 2):
            orbit_conflict[(a, b)] += N1
            if mono:
                orbit_mono_conflict[(a, b)] += N1

    print(f"  vertex_orbit links: {len(vertex_orbit_weight)}")
    print(f"  vertex co-violation links: {len(vertex_co_viol)}")
    print(f"  orbit conflict links: {len(orbit_conflict)}")

    # ---------------------------------------------------------------------------
    # Build IRDME items — SHARED dimension schema for both types
    # ---------------------------------------------------------------------------
    items = []

    # Vertex items
    for v in range(n):
        half = "half_0" if v < N1 else "half_1"
        items.append({
            "id":    f"v{v}",
            "type":  f"vertex_{half}",
            "label": f"V{v}",
            "dimensions": {
                "violation_exposure":  vcount[v],
                "structural_weight":   n - 1,          # total degree = 41 for all
                "color_balance":       abs(red_deg[v] - blue_deg[v]),
                "is_orbit":            0,
                "red_degree":          red_deg[v],
                "blue_degree":         blue_deg[v],
            }
        })

    # Orbit-type items
    for t in range(N_TYPES):
        od = orbit_data[t]
        items.append({
            "id":    f"T{t}",
            "type":  f"orbit_{orbit_class(t)}",
            "label": orbit_label(t),
            "dimensions": {
                "violation_exposure":  od["k5_mono_exposure"],
                "structural_weight":   od["n_canon_orbits"] * N1,
                "color_balance":       2 * od["color"] - 1,   # red=+1, blue=-1
                "is_orbit":            1,
                "red_degree":          0,
                "blue_degree":         0,
            }
        })

    # ---------------------------------------------------------------------------
    # Build IRDME relations
    # ---------------------------------------------------------------------------
    relations = []

    # Layer 1: vertex <-> vertex co-violation
    for (u, v), cnt in vertex_co_viol.items():
        relations.append({
            "from": f"v{u}", "to": f"v{v}",
            "type": "vertex_violation_layer",
            "directed": False,
            "dimensions": {"shared_violations": cnt}
        })

    # Layer 2: orbit <-> orbit conflict (active violations only)
    for (a, b), cnt in orbit_mono_conflict.items():
        relations.append({
            "from": f"T{a}", "to": f"T{b}",
            "type": "orbit_conflict_layer",
            "directed": False,
            "dimensions": {"shared_violations": cnt}
        })

    # Layer 3: vertex <-> orbit participation in violations
    for (v, t), cnt in vertex_orbit_weight.items():
        relations.append({
            "from": f"v{v}", "to": f"T{t}",
            "type": "vertex_orbit_layer",
            "directed": False,
            "dimensions": {"shared_violations": cnt}
        })

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    stem = Path(input_path).stem
    graph = {
        "meta": {
            "name": f"Ramsey K_{n} bipartite — {declared_v} violations",
            "description": (
                f"Bipartite IRDME network: both vertices (v0-v{n-1}) AND orbit types (T0-T{N_TYPES-1}) "
                f"are items. Three layers: vertex_violation_layer (vertex↔vertex co-violations), "
                f"orbit_conflict_layer (orbit↔orbit active conflict), "
                f"vertex_orbit_layer (vertex↔orbit: vertex appears in K5 that includes this orbit type). "
                f"Shared dimension 'violation_exposure' allows FPL across the vertex/orbit boundary."
            ),
            "domain":     "mathematics_ramsey_bipartite",
            "experiment": f"H_RAMSEY_K{n}_BIPARTITE_v1",
            "n_vertices": n,
            "n_orbit_types": N_TYPES,
            "n_violations":  declared_v,
            "layers": {
                "vertex_violation_layer": "Vertex↔vertex co-participation in K5 violations",
                "orbit_conflict_layer":   "Orbit↔orbit co-participation in ACTIVE K5 violations",
                "vertex_orbit_layer":     "Vertex↔orbit: vertex appears in K5 containing this orbit type",
            },
            "hypothesis": (
                "FPL predicts violation hubs are structurally coupled across item types: "
                "vertices with high violation_exposure bind preferentially to orbit types with "
                "high violation_exposure. v42 (vertex #1 violation hub) should link disproportionately "
                "to C12-C16 (orbit #1 violation hubs). "
                "Atlas should show vertex universal hubs co-clustering with orbit chameleon types."
            ),
        },
        "items": items,
        "relations": relations
    }

    if output_path is None:
        out = Path("examples") / f"bipartite_{stem}.json"
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(graph, f, indent=2)

    n_vv = sum(1 for r in relations if r["type"] == "vertex_violation_layer")
    n_oo = sum(1 for r in relations if r["type"] == "orbit_conflict_layer")
    n_vo = sum(1 for r in relations if r["type"] == "vertex_orbit_layer")

    print(f"\n  Saved -> {out}")
    print(f"  Items:     {len(items)} ({n} vertices + {N_TYPES} orbit types)")
    print(f"  Relations: {len(relations)}")
    print(f"    vertex_violation_layer: {n_vv}")
    print(f"    orbit_conflict_layer:   {n_oo}")
    print(f"    vertex_orbit_layer:     {n_vo}")
    print()
    print("  Run IRDME analysis:")
    print(f"    python irdme.py law   {out}")
    print(f"    python irdme.py atlas {out}")
    print(f"    python irdme.py hubs  {out}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="K_42 bi-circulant coloring JSON")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    convert(args.input, args.out)
