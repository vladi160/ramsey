"""
ramsey_hitting_set.py  --  Focused repair of K_43 near-valid coloring.

The 129 violated K_5 subgraphs form a hitting set instance:
  - Universe:   129 violated K_5s (must each be "broken")
  - Actions:    each hot edge flip (edge in >=1 violation)
  - Goal:       find a set of flips that hits all 129 violations
                without creating new ones

Strategy: iterative greedy hitting set with backtracking.
  1. Compute delta_v for every hot edge (violations fixed - violations created)
  2. Apply the best improving flip
  3. Repeat until 0 violations or no improving flip exists
  4. If stuck: try 2-flip combinations from top candidates

Usage:
  python scripts/ramsey_hitting_set.py --checkpoint datasets/RAMSEY_K43_S5_rust_checkpoint.json
"""

import json, itertools, time, argparse
from collections import defaultdict
import numpy as np

N_TARGET = 5  # avoid K_5


def edge_idx(i, j, n):
    if i > j: i, j = j, i
    return i * n - i * (i + 1) // 2 + j - i - 1


def build_state(n, assignment):
    """Build adjacency matrix and find all K_5 violations."""
    adj = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            adj[i, j] = adj[j, i] = assignment[idx]
            idx += 1

    print("  Finding violations...", end="", flush=True)
    t0 = time.time()
    violations = []
    for combo in itertools.combinations(range(n), N_TARGET):
        verts = list(combo)
        edges = [(verts[a], verts[b]) for a in range(5) for b in range(a+1, 5)]
        colors = [adj[u, v] for u, v in edges]
        if all(c == 1 for c in colors) or all(c == 0 for c in colors):
            violations.append(verts)
    print(f" {len(violations)} ({time.time()-t0:.1f}s)")
    return adj, violations


def edge_delta(adj, n, edge, violations, viol_set):
    """Compute delta violations if we flip edge (i,j)."""
    i, j = edge
    cur_color = adj[i, j]
    new_color = 1 - cur_color

    fixed = 0  # currently violated K5s that this flip fixes
    broken = 0  # currently valid K5s that this flip breaks

    # All K5s containing edge (i,j)
    others = [v for v in range(n) if v != i and v != j]
    for triple in itertools.combinations(others, 3):
        verts = sorted([i, j] + list(triple))
        key = tuple(verts)

        edges_k5 = [(verts[a], verts[b]) for a in range(5) for b in range(a+1, 5)]
        colors = [adj[u, v] for u, v in edges_k5]

        # Would this K5 be monochromatic after flip?
        new_colors = [new_color if (u == i and v == j) or (u == j and v == i)
                      else adj[u, v] for u, v in edges_k5]

        was_viol = all(c == colors[0] for c in colors)
        will_viol = all(c == new_colors[0] for c in new_colors)

        if was_viol and not will_viol:
            fixed += 1
        elif not was_viol and will_viol:
            broken += 1

    return fixed - broken, fixed, broken


def find_hot_edges(n, violations):
    hot = set()
    for verts in violations:
        for a, b in itertools.combinations(sorted(verts), 2):
            hot.add((a, b))
    return sorted(hot)


def greedy_repair(adj, n, violations, max_steps=200, beam_width=5):
    """Iterative greedy hitting set with limited backtracking."""
    adj = adj.copy()
    viol_set = set(tuple(v) for v in violations)
    history = []  # (edge, delta) for potential undo

    print(f"\n  Starting greedy repair: {len(violations)} violations, {n*(n-1)//2} edges")
    print(f"  {'step':>5}  {'violations':>12}  {'flip':>12}  {'delta':>8}  {'fixed':>7}  {'broken':>7}")

    step = 0
    while violations and step < max_steps:
        hot_edges = find_hot_edges(n, violations)
        if not hot_edges:
            print("  No hot edges — stuck")
            break

        # Evaluate all hot edges
        candidates = []
        t0 = time.time()
        for edge in hot_edges:
            d, f, b = edge_delta(adj, n, edge, violations, viol_set)
            candidates.append((d, f, b, edge))

        candidates.sort(reverse=True)  # best delta first

        best_d, best_f, best_b, best_edge = candidates[0]

        if best_d <= 0:
            print(f"  Step {step}: best flip has delta={best_d} — no improving move")
            # Try 2-flip combinations from top beam_width candidates
            print(f"  Trying 2-flip combinations from top {beam_width} candidates...")
            found_2flip = False
            top = [c[3] for c in candidates[:beam_width]]
            for e1, e2 in itertools.combinations(top, 2):
                # Simulate both flips
                adj[e1[0], e1[1]] ^= 1; adj[e1[1], e1[0]] ^= 1
                adj[e2[0], e2[1]] ^= 1; adj[e2[1], e2[0]] ^= 1
                _, viol_after = build_state(n, _adj_to_flat(adj, n))
                delta2 = len(violations) - len(viol_after)
                adj[e1[0], e1[1]] ^= 1; adj[e1[1], e1[0]] ^= 1
                adj[e2[0], e2[1]] ^= 1; adj[e2[1], e2[0]] ^= 1
                if delta2 > 0:
                    print(f"  2-flip ({e1},{e2}) delta={delta2}")
                    adj[e1[0], e1[1]] ^= 1; adj[e1[1], e1[0]] ^= 1
                    adj[e2[0], e2[1]] ^= 1; adj[e2[1], e2[0]] ^= 1
                    _, violations = build_state(n, _adj_to_flat(adj, n))
                    viol_set = set(tuple(v) for v in violations)
                    found_2flip = True
                    step += 1
                    print(f"  After 2-flip: {len(violations)} violations")
                    break
            if not found_2flip:
                print(f"  No improving 2-flip found. Stopping.")
                break
        else:
            i, j = best_edge
            adj[i, j] ^= 1; adj[j, i] ^= 1
            old_v = len(violations)
            _, violations = build_state(n, _adj_to_flat(adj, n))
            viol_set = set(tuple(v) for v in violations)
            elapsed = time.time() - t0
            print(f"  {step:>5d}  {len(violations):>12d}  {str(best_edge):>12}  {best_d:>+8d}  {best_f:>7d}  {best_b:>7d}  ({elapsed:.0f}s/step)")
            history.append((best_edge, best_d))
            step += 1

    return adj, violations


def _adj_to_flat(adj, n):
    flat = []
    for i in range(n):
        for j in range(i+1, n):
            flat.append(int(adj[i, j]))
    return flat


def run(checkpoint_path, max_steps=50):
    data = json.load(open(checkpoint_path))
    n = data['n']
    assignment = data['assignment']
    print(f"  Loaded: n={n}, declared_violations={data['violations']}")

    adj, violations = build_state(n, assignment)
    print(f"  Actual violations: {len(violations)}")

    if len(violations) == 0:
        print("  Already 0 violations!")
        return

    # Show coverage statistics
    coverage = defaultdict(int)
    hot_edges = find_hot_edges(n, violations)
    for verts in violations:
        for a, b in itertools.combinations(sorted(verts), 2):
            coverage[(a, b)] += 1

    top_cover = sorted(coverage.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Hot edges: {len(hot_edges)}, Cold edges: {n*(n-1)//2 - len(hot_edges)}")
    print(f"  Top edges by violation coverage:")
    print(f"  {'edge':>12}  {'coverage':>10}  {'current color':>14}")
    for (a, b), cnt in top_cover:
        color = 'red' if adj[a, b] == 1 else 'blue'
        print(f"  ({a:2d},{b:2d}):      {cnt:>10d}  {color:>14}")

    adj_repaired, violations_after = greedy_repair(adj, n, violations, max_steps=max_steps)

    print(f"\n  Final violations: {len(violations_after)}")
    if len(violations_after) == 0:
        print("  *** VALID COLORING FOUND! R(5,5) >= 44 PROVEN! ***")
        flat = _adj_to_flat(adj_repaired, n)
        out = {
            'n': n, 's': 5, 'violations': 0,
            'restart': 0, 'total_iters': 0, 'snap_count': 0,
            'timestamp': time.strftime('%Y-%m-%d'),
            'assignment': flat
        }
        out_path = 'datasets/RAMSEY_K43_S5_VALID.json'
        json.dump(out, open(out_path, 'w'), indent=2)
        print(f"  Saved to {out_path}")
    else:
        print(f"  Improvement: {data['violations']} -> {len(violations_after)} violations")
        if len(violations_after) < data['violations']:
            flat = _adj_to_flat(adj_repaired, n)
            out = dict(data)
            out['violations'] = len(violations_after)
            out['assignment'] = flat
            out_path = checkpoint_path.replace('.json', f'_repaired_{len(violations_after)}v.json')
            json.dump(out, open(out_path, 'w'), indent=2)
            print(f"  Saved to {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='datasets/RAMSEY_K43_S5_rust_checkpoint.json')
    ap.add_argument('--steps', type=int, default=50)
    args = ap.parse_args()
    run(args.checkpoint, max_steps=args.steps)
