"""
ramsey_to_irdme.py  --  Convert a Ramsey coloring JSON to IRDME graph.json format.

Usage:
    python scripts/ramsey_to_irdme.py datasets/RAMSEY_K42_S5_bicirculant_84v.json
    python scripts/ramsey_to_irdme.py datasets/RAMSEY_K43_S5_rust_checkpoint.json

What it builds (three layers):
    red_layer   -- edges colored red  (color=1)
    blue_layer  -- edges colored blue (color=0)
    violation   -- edges where both endpoints co-appear in >=1 monochromatic K_5

Node dimensions:
    red_degree        -- degree in red layer
    blue_degree       -- degree in blue layer
    violation_count   -- number of monochromatic K_5s this vertex appears in

Then run any IRDME command on the output, e.g.:
    python irdme.py law    examples/ramsey_K42_84v.json
    python irdme.py atlas  examples/ramsey_K42_84v.json
    python irdme.py hubs   examples/ramsey_K42_84v.json
    python irdme.py inspect examples/ramsey_K42_84v.json
    python irdme.py benchmark examples/ramsey_K42_84v.json
"""

import json, itertools, time, sys, argparse
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# K_5 detection -- find all monochromatic 5-cliques
# ---------------------------------------------------------------------------

def find_violations(n, adj):
    """
    adj[u][v] = 0 (blue) or 1 (red), u != v.
    Returns list of (color, [v0,v1,v2,v3,v4]) for each monochromatic K_5.
    """
    violations = []
    for combo in itertools.combinations(range(n), 5):
        edges = [(combo[i], combo[j])
                 for i in range(5) for j in range(i+1, 5)]
        colors = [adj[u][v] for u, v in edges]
        if all(c == 1 for c in colors):
            violations.append((1, list(combo)))   # red K_5
        elif all(c == 0 for c in colors):
            violations.append((0, list(combo)))   # blue K_5
    return violations


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_path: str, output_path: str | None = None):
    data = json.load(open(input_path))
    n = data["n"]
    s = data.get("s", 5)
    v_total = data["violations"]
    assignment = data["assignment"]  # flat list, edge (i,j) i<j order

    # Build adjacency matrix
    adj = [[None] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = assignment[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1

    print(f"  Loaded: n={n}, s={s}, declared_violations={v_total}")
    print(f"  Finding all monochromatic K_5 subgraphs... ", end="", flush=True)
    t0 = time.time()
    violations = find_violations(n, adj)
    print(f"{len(violations)} found  ({time.time()-t0:.1f}s)")

    # Per-vertex violation count
    vcount = [0] * n
    # Per-pair co-violation count
    co_viol = defaultdict(int)
    for color, verts in violations:
        for v in verts:
            vcount[v] += 1
        for u, v in itertools.combinations(sorted(verts), 2):
            co_viol[(u, v)] += 1

    # Per-vertex degree in each color
    red_deg  = [sum(1 for j in range(n) if j != i and adj[i][j] == 1) for i in range(n)]
    blue_deg = [sum(1 for j in range(n) if j != i and adj[i][j] == 0) for i in range(n)]

    print(f"  Violation counts per vertex: min={min(vcount)}  max={max(vcount)}  "
          f"mean={sum(vcount)/n:.1f}")
    print(f"  Red degree:  min={min(red_deg)}  max={max(red_deg)}")
    print(f"  Blue degree: min={min(blue_deg)}  max={max(blue_deg)}")

    # Half classification (only meaningful for K_42 bi-circulant; for unconstrained label "free")
    def half_label(v):
        if n == 42 or n == 43:
            if v == 42:
                return "extra_vertex"
            return "half_0" if v < 21 else "half_1"
        return "vertex"

    # ---------------------------------------------------------------------------
    # Build IRDME items
    # ---------------------------------------------------------------------------
    items = []
    for v in range(n):
        items.append({
            "id": f"v{v}",
            "type": half_label(v),
            "label": f"V{v}",
            "dimensions": {
                "red_degree":      red_deg[v],
                "blue_degree":     blue_deg[v],
                "violation_count": vcount[v],
                "position":        v
            }
        })

    # ---------------------------------------------------------------------------
    # Build IRDME relations (3 layers)
    # ---------------------------------------------------------------------------
    relations = []

    # Layer 1 & 2: red / blue edges
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = assignment[idx]
            layer = "red_layer" if c == 1 else "blue_layer"
            relations.append({
                "from": f"v{i}",
                "to":   f"v{j}",
                "type": layer,
                "directed": False,
                "dimensions": {"color": c}
            })
            idx += 1

    # Layer 3: violation co-participation graph
    for (u, v), cnt in co_viol.items():
        relations.append({
            "from": f"v{u}",
            "to":   f"v{v}",
            "type": "violation_layer",
            "directed": False,
            "dimensions": {"shared_violations": cnt}
        })

    # ---------------------------------------------------------------------------
    # Meta block with experiment hypothesis
    # ---------------------------------------------------------------------------
    stem = Path(input_path).stem
    graph = {
        "meta": {
            "name": f"Ramsey K_{n} coloring — {len(violations)} violations",
            "description": (
                f"2-coloring of the complete graph K_{n} (n={n} vertices, "
                f"n*(n-1)/2={n*(n-1)//2} edges) with {len(violations)} monochromatic K_{s} violations. "
                f"Three layers: red_layer (edges colored red), blue_layer (edges colored blue), "
                f"violation_layer (pairs that co-appear in >=1 monochromatic K_{s}). "
                f"Node dimensions: red_degree, blue_degree, violation_count. "
                f"Source: Ramsey R(5,5) computational search, Ivanov 2026."
            ),
            "domain": "mathematics_ramsey",
            "experiment": f"H_RAMSEY_K{n}_STRUCTURE_v1",
            "source_file": str(input_path),
            "n_vertices": n,
            "n_edges": n * (n - 1) // 2,
            "n_violations": len(violations),
            "declared_violations": v_total,
            "layers": {
                "red_layer":       f"Edges colored red in the 2-coloring of K_{n}",
                "blue_layer":      f"Edges colored blue (complement of red layer)",
                "violation_layer": f"Pairs of vertices that co-appear in >=1 monochromatic K_{s}"
            },
            "hypothesis": (
                "FPL predicts that structural hub rank in the red_layer (or blue_layer) "
                "correlates with hub rank in the violation_layer — vertices structurally "
                "prominent in one color layer are disproportionately involved in violations. "
                "Additionally, irdme atlas should reveal distinct structural classes among "
                "the vertices, with violation_count separating the classes."
            ),
            "prediction": (
                "Top structural hubs (irdme atlas 'hub zone') will have violation_count "
                f"> 2x the mean ({sum(vcount)/n:.1f}). "
                "Spearman correlation between red_degree and violation_count should be "
                "positive and significant (p < 0.05)."
            )
        },
        "items": items,
        "relations": relations
    }

    # Output path
    if output_path is None:
        out = Path("examples") / f"ramsey_{stem}.json"
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"\n  Saved -> {out}")
    print(f"  Items:     {len(items)}")
    print(f"  Relations: {len(relations)}  "
          f"(red={sum(1 for r in relations if r['type']=='red_layer')}, "
          f"blue={sum(1 for r in relations if r['type']=='blue_layer')}, "
          f"violation={sum(1 for r in relations if r['type']=='violation_layer')})")
    print()
    print("  Run IRDME analysis:")
    print(f"    python irdme.py inspect   {out}")
    print(f"    python irdme.py law       {out}")
    print(f"    python irdme.py hubs      {out}")
    print(f"    python irdme.py atlas     {out}")
    print(f"    python irdme.py benchmark {out}")
    print(f"    python irdme.py null      {out}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input",  help="Ramsey coloring JSON (datasets/...)")
    ap.add_argument("--out",  default=None, help="Output path (default: examples/ramsey_<stem>.json)")
    args = ap.parse_args()
    convert(args.input, args.out)
