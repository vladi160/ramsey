"""
ramsey_orbits_to_irdme.py  --  IRDME dataset where ORBIT TYPES are items.

Design rationale:
  The tabu search operates on orbit types (41 bits for K_42 bi-circulant),
  not on vertices. The vertex-as-item IRDME design shows which vertices are
  structurally stressed, but the orbit-as-item design shows which DECISIONS
  (edge-type assignments) are most pivotal in the violation landscape.

Items (41 for K_42 bi-circulant):
  T0..T9   = A1..A10 (within half-0, position diff k)
  T10..T19 = B1..B10 (within half-1, position diff k)
  T20..T40 = C0..C20 (between halves, relative position diff k)

Dimensions per orbit type:
  color             -- current assignment (0=blue, 1=red)
  k5_exposure       -- number of canonical K5 orbits containing this type
  k5_mono_exposure  -- how many of those K5 orbits are currently monochromatic
  delta_if_flip     -- change in violations if this type were flipped (negative = improvement)
  orbit_size        -- always 21 for K_42 Z_21 bi-circulant

Layers:
  conflict_layer    -- two types connected if they co-appear in >=1 K5 orbit
                       (flipping one affects the other's violation context)
  mono_conflict_layer -- same but only monochromatic K5 orbits (active violations)

FPL question:
  Do types with high k5_mono_exposure (many active violations) also have
  high |delta_if_flip| (high flip impact)? If yes, flip-impact predicts
  which types to target in search.

Usage:
  python scripts/ramsey_orbits_to_irdme.py datasets/RAMSEY_K42_S5_bicirculant_84v.json
  python irdme.py law   examples/orbits_RAMSEY_K42_S5_bicirculant_84v.json
  python irdme.py atlas examples/orbits_RAMSEY_K42_S5_bicirculant_84v.json
  python irdme.py hubs  examples/orbits_RAMSEY_K42_S5_bicirculant_84v.json
"""

import json, itertools, time, sys, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------------------
# Z_21 orbit utilities (mirror of ramsey_symmetry.py)
# ---------------------------------------------------------------------------

N     = 42
N1    = 21
N_TYPES = 41

def vertex_decompose(v):
    return v % N1, v // N1   # (pos, half)

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
    if t < 10:
        return f"A{t+1}"
    elif t < 20:
        return f"B{t-9}"
    else:
        return f"C{t-20}"

def orbit_class(t):
    if t < 10:
        return "within_half0"
    elif t < 20:
        return "within_half1"
    else:
        return "cross_half"

# ---------------------------------------------------------------------------
# Precompute canonical K5 orbits and their type masks
# ---------------------------------------------------------------------------

def build_k5_orbits():
    """
    Enumerate all C(42,5) five-tuples, take canonical representative
    (lexicographically smallest in its Z_21 orbit), collect unique ones.
    For each canonical tuple, record the 41-bit mask of which edge types it uses.
    """
    print("  Building canonical K5 orbits...", end="", flush=True)
    t0 = time.time()
    seen = {}
    for combo in itertools.combinations(range(N), 5):
        # Canonical = lexicographically smallest rotation
        min_rep = combo
        for k in range(1, N1):
            rotated = tuple(sorted(((v % N1 + k) % N1 + (v // N1) * N1) for v in combo))
            if rotated < min_rep:
                min_rep = rotated
        if min_rep not in seen:
            mask = 0
            for i in range(5):
                for j in range(i+1, 5):
                    t = edge_orbit(min_rep[i], min_rep[j])
                    mask |= (1 << t)
            seen[min_rep] = mask
    canon_list = list(seen.items())  # [(tuple, mask), ...]
    print(f" {len(canon_list)} orbits  ({time.time()-t0:.1f}s)")
    return canon_list

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(input_path, output_path=None):
    data = json.load(open(input_path))
    n = data["n"]
    if n != N:
        print(f"  Warning: this script is designed for K_42 (n=42), got n={n}")

    assignment_flat = data["assignment"]  # 861 ints, edge (i,j) i<j order
    declared_v = data["violations"]

    # Recover orbit type assignments from flat assignment
    types = [0] * N_TYPES
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            t = edge_orbit(i, j)
            types[t] = assignment_flat[idx]
            idx += 1

    assignment_int = sum(int(types[t]) << t for t in range(N_TYPES))
    print(f"  Loaded: n={n}, declared_violations={declared_v}")
    print(f"  Type assignments: {sum(types)} red types, {N_TYPES-sum(types)} blue types")

    # Build canonical K5 orbits
    canon_list = build_k5_orbits()

    # For each canon orbit, check monochromaticity
    n_orbits = len(canon_list)
    masks_np  = np.array([m for _, m in canon_list], dtype=np.int64)
    mults_np  = np.ones(n_orbits, dtype=np.int32) * N1  # each orbit has size 21

    def count_violations_exact(asgn_int):
        """Count monochromatic K5 orbits × 21."""
        total = 0
        for mask in masks_np:
            bits = asgn_int & mask
            if bits == 0 or bits == mask:
                total += N1
        return total

    actual_v = count_violations_exact(assignment_int)
    print(f"  Actual violations: {actual_v} (declared: {declared_v})")

    # Build type_to_canons: for each type t, which canon orbit indices contain it
    type_to_canons = [[] for _ in range(N_TYPES)]
    for ci, (_, mask) in enumerate(canon_list):
        for t in range(N_TYPES):
            if mask >> t & 1:
                type_to_canons[t].append(ci)

    # Per-type: k5_exposure, k5_mono_exposure, delta_if_flip
    def is_mono(ci, asgn_int):
        mask = int(masks_np[ci])
        bits = asgn_int & mask
        return bits == 0 or bits == mask

    type_data = []
    for t in range(N_TYPES):
        canons = type_to_canons[t]
        k5_exp = len(canons) * N1          # total K5 subgraphs containing this type
        k5_mono = sum(N1 for ci in canons if is_mono(ci, assignment_int))

        # Delta if we flip type t
        flipped_int = assignment_int ^ (1 << t)
        delta_v = count_violations_exact(flipped_int) - actual_v
        type_data.append({
            "color":           types[t],
            "k5_exposure":     k5_exp,
            "k5_mono_exposure": k5_mono,
            "delta_if_flip":   delta_v,
            "orbit_size":      N1,
            "n_canon_orbits":  len(canons),
        })

    print(f"  Per-type stats computed.")

    # Build conflict graph between types: share >= 1 canon K5 orbit
    conflict = defaultdict(int)      # (t1,t2) -> n shared K5 orbits
    mono_conflict = defaultdict(int) # only monochromatic ones

    for ci, (_, mask) in enumerate(canon_list):
        involved = [t for t in range(N_TYPES) if mask >> t & 1]
        mono = is_mono(ci, assignment_int)
        for a, b in itertools.combinations(involved, 2):
            conflict[(a,b)] += N1
            if mono:
                mono_conflict[(a,b)] += N1

    print(f"  Conflict edges: {len(conflict)} total, {len(mono_conflict)} mono-active")

    # ---------------------------------------------------------------------------
    # Build IRDME items
    # ---------------------------------------------------------------------------
    items = []
    for t in range(N_TYPES):
        d = type_data[t]
        items.append({
            "id":    f"T{t}",
            "type":  orbit_class(t),
            "label": orbit_label(t),
            "dimensions": {
                "color":              d["color"],
                "k5_exposure":        d["k5_exposure"],
                "k5_mono_exposure":   d["k5_mono_exposure"],
                "delta_if_flip":      d["delta_if_flip"],
                "n_canon_orbits":     d["n_canon_orbits"],
            }
        })

    # ---------------------------------------------------------------------------
    # Build IRDME relations
    # ---------------------------------------------------------------------------
    relations = []

    for (t1, t2), cnt in conflict.items():
        relations.append({
            "from":     f"T{t1}",
            "to":       f"T{t2}",
            "type":     "conflict_layer",
            "directed": False,
            "dimensions": {"shared_k5_subgraphs": cnt}
        })

    for (t1, t2), cnt in mono_conflict.items():
        relations.append({
            "from":     f"T{t1}",
            "to":       f"T{t2}",
            "type":     "mono_conflict_layer",
            "directed": False,
            "dimensions": {"shared_violations": cnt}
        })

    # ---------------------------------------------------------------------------
    # Meta + output
    # ---------------------------------------------------------------------------
    stem = Path(input_path).stem
    graph = {
        "meta": {
            "name": f"Ramsey K_{n} orbit types — {declared_v} violations",
            "description": (
                f"IRDME dataset where ORBIT TYPES are items (not vertices). "
                f"K_{n} Z_21 bi-circulant: {N_TYPES} orbit types x {N1} edges each = {N*(N-1)//2} edges. "
                f"Each item is one orbit type; dimensions encode its role in violations. "
                f"conflict_layer: two types share a K5 subgraph (structural dependency). "
                f"mono_conflict_layer: two types share an ACTIVE violation. "
                f"Source: Ramsey R(5,5) computational search, Ivanov 2026."
            ),
            "domain":      "mathematics_ramsey_orbits",
            "experiment":  f"H_RAMSEY_K{n}_ORBITS_v1",
            "source_file": str(input_path),
            "n_types":     N_TYPES,
            "n_violations": actual_v,
            "layers": {
                "conflict_layer":      "Two orbit types share >= 1 K5 subgraph",
                "mono_conflict_layer": "Two orbit types share >= 1 ACTIVE (monochromatic) violation",
            },
            "hypothesis": (
                "FPL predicts that orbit types with high k5_mono_exposure (many active violations) "
                "also have high |delta_if_flip| (high flip impact). "
                "If true: hubs in mono_conflict_layer are the best edges to flip next. "
                "Atlas should reveal classes of orbit types by structural role: "
                "bridge types (C_k), intra-half types (A_k, B_k) may show different hub profiles."
            ),
            "prediction": (
                "Spearman correlation between k5_mono_exposure and |delta_if_flip| should be "
                "positive and significant. Cross-half types (C_k) are expected to be hubs "
                "in conflict_layer (they connect the two halves and appear in more K5 orbits)."
            )
        },
        "items": items,
        "relations": relations
    }

    if output_path is None:
        out = Path("examples") / f"orbits_{stem}.json"
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"\n  Saved -> {out}")
    print(f"  Items:     {len(items)} orbit types")
    print(f"  Relations: {len(relations)}")
    print(f"    conflict_layer:      {sum(1 for r in relations if r['type']=='conflict_layer')}")
    print(f"    mono_conflict_layer: {sum(1 for r in relations if r['type']=='mono_conflict_layer')}")
    print()

    # Print top types by k5_mono_exposure
    ranked = sorted(range(N_TYPES), key=lambda t: -type_data[t]["k5_mono_exposure"])
    print(f"  Top 10 types by active violation exposure:")
    print(f"  {'type':>5}  {'label':>5}  {'class':>15}  {'color':>5}  {'k5_mono':>8}  {'delta':>8}")
    for t in ranked[:10]:
        d = type_data[t]
        print(f"  T{t:>3d}   {orbit_label(t):>5}  {orbit_class(t):>15}  {'red' if d['color'] else 'blue':>5}  "
              f"{d['k5_mono_exposure']:>8}  {d['delta_if_flip']:>+8}")

    print()
    print("  Run IRDME analysis:")
    print(f"    python irdme.py law   {out}")
    print(f"    python irdme.py atlas {out}")
    print(f"    python irdme.py hubs  {out}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input",  help="K_42 bi-circulant coloring JSON")
    ap.add_argument("--out",  default=None)
    args = ap.parse_args()
    convert(args.input, args.out)
