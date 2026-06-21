"""
Ramsey SAT Encoder + Solver interface.

Encodes the Ramsey(s,s;n) problem as DIMACS CNF:
  "Does K_n have a valid 2-coloring with no monochromatic K_s?"
  SATISFIABLE  ⟺  R(s,s) > n
  UNSATISFIABLE ⟺  R(s,s) ≤ n

Variable x_{i,j} = 1 (red edge), 0 (blue edge), for all i < j.
Clauses (for each s-clique C with edges e1..ek, k=C(s,2)):
  NOT all-red:  (-e1 ∨ -e2 ∨ ... ∨ -ek)
  NOT all-blue: ( e1 ∨  e2 ∨ ... ∨  ek)

Usage:
  python scripts/ramsey_sat.py --n 17 --s 4                     # encode only
  python scripts/ramsey_sat.py --n 17 --s 4 --solve             # encode + call kissat
  python scripts/ramsey_sat.py --n 42 --s 5 --solve             # K_42 (hard, hours)
  python scripts/ramsey_sat.py --n 42 --s 5 --solve --timeout 3600

Install Kissat (fast CDCL SAT solver):
  Windows: https://github.com/arminbiere/kissat/releases
           Download kissat.exe, put in PATH or same directory
  Linux:   sudo apt install kissat   OR   ./configure && make
  Run:     kissat problem.cnf
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np


def edge_var(i: int, j: int, n: int) -> int:
    """1-indexed SAT variable for edge (i,j) with i < j."""
    if i > j:
        i, j = j, i
    # Map (i,j) to 1..C(n,2)
    # Use: var = (i * (2n - i - 1)) // 2 + (j - i)
    return (i * (2 * n - i - 1)) // 2 + (j - i)


def build_cnf(
    n: int,
    s: int,
    symmetry_break: bool = True,
) -> tuple[list[list[int]], int]:
    """
    Build DIMACS clauses for Ramsey(s,s;n).
    Returns (clauses, num_vars).

    Symmetry breaking (dramatically reduces search space):
    - Fix edge (0,1) = red (valid by edge-transitivity of K_n)
    - Order vertex 0's neighbors: if (0,j) is red then (0,j-1) is red
      → WLOG vertex 0's red neighbors are a prefix {1,...,k}
    - Fix red/blue symmetry: vertex 0's red degree ≤ (n-1)/2
      → WLOG more red neighbors than blue is forbidden
    These reductions are valid because any valid coloring of K_n can be
    relabeled to satisfy all three without loss of generality.
    """
    num_vars = n * (n - 1) // 2
    clauses: list[list[int]] = []

    for clique in combinations(range(n), s):
        clique_edges = list(combinations(clique, 2))
        vars_ = [edge_var(i, j, n) for i, j in clique_edges]
        # NOT all-red
        clauses.append([-v for v in vars_])
        # NOT all-blue
        clauses.append([v for v in vars_])

    if symmetry_break:
        # (1) Fix edge (0,1) = red
        clauses.append([edge_var(0, 1, n)])

        # (2) Ordering: vertex 0's red neighbors form a prefix
        #     (0,j) is red → (0,j-1) is red, for j=2..n-1
        #     Encoded as: NOT(0,j)=red OR (0,j-1)=red
        #     i.e., [-var(0,j), var(0,j-1)]
        for j in range(2, n):
            clauses.append([-edge_var(0, j, n), edge_var(0, j - 1, n)])

        # (3) Red/blue symmetry: vertex 0 has ≤ floor((n-1)/2) red neighbors
        #     Enforce: edge (0, floor((n-1)/2)+1) must be blue
        #     Combined with the prefix ordering, this means red_degree(0) ≤ half
        mid = (n - 1) // 2  # last allowed red-neighbor index in {1..n-1}
        if mid + 1 < n:
            clauses.append([-edge_var(0, mid + 1, n)])  # edge (0, mid+1) = blue

    return clauses, num_vars


def write_dimacs(clauses: list[list[int]], num_vars: int, path: str) -> None:
    """Write CNF in standard DIMACS format."""
    with open(path, "w") as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def parse_solution(output: str, n: int) -> np.ndarray | None:
    """Parse SAT solver output into adjacency matrix (red=1)."""
    adj = np.zeros((n, n), dtype=np.int8)
    found_sat = False
    vals: dict[int, bool] = {}

    for line in output.splitlines():
        line = line.strip()
        if line == "s SATISFIABLE" or line.startswith("s SAT"):
            found_sat = True
        elif line.startswith("v "):
            for tok in line[2:].split():
                v = int(tok)
                if v == 0:
                    break
                vals[abs(v)] = (v > 0)

    if not found_sat:
        return None

    for i in range(n):
        for j in range(i + 1, n):
            var = edge_var(i, j, n)
            if vals.get(var, False):
                adj[i, j] = 1
                adj[j, i] = 1

    return adj


def verify_coloring(adj: np.ndarray, s: int, n: int) -> tuple[bool, str]:
    """Check no monochromatic K_s exists."""
    adj_blue = 1 - adj
    np.fill_diagonal(adj_blue, 0)

    def find_clique(a: np.ndarray, sz: int) -> list[int] | None:
        def ext(clique, cands):
            if len(clique) == sz:
                return clique
            need = sz - len(clique)
            for i in range(len(cands) - need + 1):
                v = cands[i]
                nc = [c for c in cands[i + 1:] if a[v, c]]
                if 1 + len(nc) >= need:
                    r = ext(clique + [v], nc)
                    if r:
                        return r
            return None

        for start in range(n):
            cands = [j for j in range(start + 1, n) if a[start, j]]
            r = ext([start], cands)
            if r:
                return r
        return None

    red_c = find_clique(adj, s)
    if red_c:
        return False, f"Red K_{s} found: {red_c}"
    blue_c = find_clique(adj_blue, s)
    if blue_c:
        return False, f"Blue K_{s} found: {blue_c}"
    return True, f"Valid: no monochromatic K_{s}"


def save_coloring(adj: np.ndarray, n: int, s: int, out_path: str) -> None:
    items = [{"id": str(i), "type": "vertex", "dimensions": {"index": i}} for i in range(n)]
    rels = []
    for i in range(n):
        for j in range(i + 1, n):
            rels.append({
                "from": str(i), "to": str(j),
                "type": "red" if adj[i, j] else "blue",
                "directed": False
            })
    data = {
        "meta": {
            "name": f"Ramsey K_{n} valid 2-coloring, avoid K_{s}",
            "domain": "mathematics", "subdomain": "ramsey_theory",
            "n": n, "s": s,
            "result": f"R({s},{s}) > {n}"
        },
        "items": items,
        "relations": rels
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved coloring -> {out_path}")


def find_kissat() -> str | None:
    """Locate kissat binary."""
    for name in ["kissat", "kissat.exe"]:
        # Check PATH
        try:
            r = subprocess.run(["where" if sys.platform == "win32" else "which", name],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return name
        except FileNotFoundError:
            pass
        # Check current directory
        local = Path(".") / name
        if local.exists():
            return str(local)
        # Check scripts directory
        scripts = Path("scripts") / name
        if scripts.exists():
            return str(scripts)
    return None


def main():
    parser = argparse.ArgumentParser(description="Ramsey SAT encoder")
    parser.add_argument("--n", type=int, required=True, help="Graph size (K_n)")
    parser.add_argument("--s", type=int, default=5, help="Clique size to avoid (default: 5)")
    parser.add_argument("--solve", action="store_true", help="Run kissat after encoding")
    parser.add_argument("--solver", default="kissat", help="SAT solver binary (default: kissat)")
    parser.add_argument("--timeout", type=int, default=None, help="Solver timeout in seconds")
    parser.add_argument("--cnf", default=None, help="Output .cnf path (default: auto)")
    parser.add_argument("--out", default=None, help="Output JSON path for solution")
    parser.add_argument("--no-sym", action="store_true", help="Disable symmetry breaking clauses")
    args = parser.parse_args()

    n, s = args.n, args.s
    num_edges = n * (n - 1) // 2
    from math import comb
    num_cliques = comb(n, s)

    print()
    print("=" * 70)
    print(f"  RAMSEY SAT ENCODER: K_{n}, avoiding monochromatic K_{s}")
    print(f"  Variables: {num_edges} edges")
    print(f"  Cliques:   {num_cliques} x 2 = {num_cliques * 2} clauses")
    print("=" * 70)
    print()

    sym = not args.no_sym
    t0 = time.time()
    print(f"  Symmetry breaking: {'enabled' if sym else 'disabled'}")
    print("  Building CNF...", end=" ", flush=True)
    clauses, num_vars = build_cnf(n, s, symmetry_break=sym)
    print(f"done ({time.time()-t0:.1f}s)")

    cnf_path = args.cnf or f"datasets/RAMSEY_K{n}_S{s}.cnf"
    Path(cnf_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {cnf_path}...", end=" ", flush=True)
    write_dimacs(clauses, num_vars, cnf_path)
    size_mb = Path(cnf_path).stat().st_size / 1e6
    print(f"done ({size_mb:.1f} MB)")

    print()
    print(f"  CNF saved. To solve:")
    print(f"    kissat {cnf_path}")
    print()

    if not args.solve:
        return

    solver = find_kissat() or args.solver
    print(f"  Searching for solver '{solver}'...")

    cmd = [solver, cnf_path]
    if args.timeout:
        cmd += ["--time", str(args.timeout)]

    print(f"  Running: {' '.join(cmd)}")
    print(f"  (This may take minutes to hours for large instances)")
    print()

    t1 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout + 60 if args.timeout else None
        )
        solver_output = result.stdout + result.stderr
    except FileNotFoundError:
        print(f"  ERROR: Solver '{solver}' not found.")
        print()
        print("  Install Kissat:")
        print("    Windows: https://github.com/arminbiere/kissat/releases")
        print("             Download kissat.exe, place in PATH or project root")
        print("    Linux:   sudo apt install kissat")
        print("             OR: git clone https://github.com/arminbiere/kissat")
        print("                 cd kissat && ./configure && make")
        print("    Then run:")
        print(f"      kissat {cnf_path}")
        return
    except subprocess.TimeoutExpired:
        print(f"  Solver timed out after {args.timeout}s")
        return

    elapsed = time.time() - t1
    print(solver_output)
    print(f"  Solver finished in {elapsed:.1f}s")
    print()

    # Parse result
    if "SATISFIABLE" in solver_output and "UNSATISFIABLE" not in solver_output:
        print(f"  RESULT: SATISFIABLE — valid K_{n} coloring exists!")
        print(f"  This proves R({s},{s}) > {n}")
        adj = parse_solution(solver_output, n)
        if adj is not None:
            valid, msg = verify_coloring(adj, s, n)
            print(f"  Verification: {msg}")
            if valid:
                out_path = args.out or f"datasets/RAMSEY_K{n}_S{s}_solution.json"
                save_coloring(adj, n, s, out_path)
        else:
            print("  (Could not parse variable assignments from solver output)")
    elif "UNSATISFIABLE" in solver_output:
        print(f"  RESULT: UNSATISFIABLE — no valid K_{n} coloring exists!")
        print(f"  This proves R({s},{s}) <= {n}")
    else:
        print("  RESULT: Unknown (solver may have timed out or output is non-standard)")

    print()


if __name__ == "__main__":
    main()
