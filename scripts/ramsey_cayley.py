"""
ramsey_cayley.py — Non-cyclic Cayley graph coloring search for R(5,5)

Groups of order 42 not covered by the Z_42 exhaustive search:
  d21  : D_21 = Z_21 rtimes Z_2   (31 orbit vars: 10 pairs + 21 singletons)
  f42  : F_42 = Z_7 rtimes Z_6    (24 orbit vars: 17 pairs +  7 singletons)
  z7s3 : Z_7 x S_3            (22 orbit vars: 19 pairs +  3 singletons)

A Cayley 2-coloring on G: edge {g,h} is red iff g^-1h ∈ S (connection set).
Left-translation symmetry reduces 850,668 K5s to ~20,254 canonical orbits.
Numpy bitmask violation counting — same technique as ramsey_symmetry.py.

Usage:
  python scripts/ramsey_cayley.py --group d21  --steps 100000 --tabu 8 --seed 42
  python scripts/ramsey_cayley.py --group f42  --steps 100000 --tabu 8 --seed 42
  python scripts/ramsey_cayley.py --group z7s3 --steps 100000 --tabu 8 --seed 42
"""

import sys, json, time, argparse, random
from itertools import combinations
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

N = 42  # vertices = group order


# ─── Group constructions ──────────────────────────────────────────────────────

def build_d21():
    """D_21 = <r,s | r^21=s^2=e, srs^-1=r^-1>
    Element j = 21*s + k  (k∈Z_21, s∈{0,1}).  Identity = 0.
    Product: (k1,s1)*(k2,s2) = ((k1 + (−1)^s1 * k2) mod 21, s1xors2)
    """
    mul = np.empty((42, 42), dtype=np.int8)
    for i in range(42):
        k1, s1 = i % 21, i >> 5 & 0  # simpler:
        k1, s1 = i % 21, i // 21
        for j in range(42):
            k2, s2 = j % 21, j // 21
            mul[i, j] = 21 * (s1 ^ s2) + (k1 + (1 - 2*s1) * k2) % 21
    inv = np.empty(42, dtype=np.int8)
    for i in range(42):
        k, s = i % 21, i // 21
        inv[i] = 21 * s + (-k if s == 0 else k) % 21
    return mul, inv, "D_21 = Z_21 rtimes Z_2"


def build_f42():
    """F_42 = Z_7 rtimes Z_6  (Frobenius group)
    Element j = 7*b + a  (a∈Z_7, b∈Z_6).  Identity = 0.
    Product: (a1,b1)*(a2,b2) = (a1 + 3^b1·a2  mod 7,  b1+b2 mod 6)
    3 is a primitive root mod 7: 3^0..5 = 1,3,2,6,4,5
    """
    pw3 = [pow(3, b, 7) for b in range(6)]
    mul = np.empty((42, 42), dtype=np.int8)
    for i in range(42):
        a1, b1 = i % 7, i // 7
        for j in range(42):
            a2, b2 = j % 7, j // 7
            mul[i, j] = 7 * ((b1 + b2) % 6) + (a1 + pw3[b1] * a2) % 7
    inv = np.empty(42, dtype=np.int8)
    for i in range(42):
        for j in range(42):
            if mul[i, j] == 0:
                inv[i] = j
                break
    return mul, inv, "F_42 = Z_7 rtimes Z_6 (Frobenius)"


def build_z7s3():
    """Z_7 x S_3
    S_3 elements 0..5: e,r,r²,s,sr,sr²  as permutations of {0,1,2}
    Element j = 7*t + a  (a∈Z_7, t∈{0..5}).  Identity = 0.
    Product: (a1,t1)*(a2,t2) = (a1+a2 mod 7, t1·t2)
    """
    perms = [[0,1,2],[1,2,0],[2,0,1],[0,2,1],[2,1,0],[1,0,2]]
    s3inv = [0, 2, 1, 3, 4, 5]
    s3mul = [[0]*6 for _ in range(6)]
    for i in range(6):
        for j in range(6):
            comp = [perms[i][perms[j][x]] for x in range(3)]
            for k in range(6):
                if perms[k] == comp:
                    s3mul[i][j] = k
                    break
    mul = np.empty((42, 42), dtype=np.int8)
    for i in range(42):
        a1, t1 = i % 7, i // 7
        for j in range(42):
            a2, t2 = j % 7, j // 7
            mul[i, j] = 7 * s3mul[t1][t2] + (a1 + a2) % 7
    inv = np.empty(42, dtype=np.int8)
    for i in range(42):
        a, t = i % 7, i // 7
        inv[i] = 7 * s3inv[t] + (-a) % 7
    return mul, inv, "Z_7 x S_3"


GROUPS = {'d21': build_d21, 'f42': build_f42, 'z7s3': build_z7s3}


# ─── Orbit structure ──────────────────────────────────────────────────────────

def find_orbits(inv_table):
    """Partition non-identity elements into inverse-closed pairs/singletons."""
    seen, orbits = set(), []
    for i in range(1, 42):
        if i in seen:
            continue
        j = int(inv_table[i])
        orb = frozenset({i, j})
        seen |= orb
        orbits.append(orb)
    return orbits


def make_orbit_of(orbits):
    orbit_of = [-1] * 42
    for oi, orb in enumerate(orbits):
        for e in orb:
            orbit_of[e] = oi
    return orbit_of


# ─── K5 canonical lookup ──────────────────────────────────────────────────────

def build_lookup(mul_table, inv_table, orbit_of, n_orbits):
    """Canonicalise 850,668 K5s under left-translation → ~20k canonical orbits.
    Returns masks, mults, type_to_canons arrays for numpy bitmask counting.
    """
    print("  Canonicalising K5 orbits under left translation...", end="", flush=True)
    t0 = time.time()

    inv = inv_table  # int8 array
    canon_map = {}   # canonical tuple → [multiplicity, bitmask]

    for combo in combinations(range(42), 5):
        best = None
        for v in combo:
            h = int(inv[v])
            cand = tuple(sorted(int(mul_table[h, w]) for w in combo))
            if best is None or cand < best:
                best = cand
        entry = canon_map.get(best)
        if entry is not None:
            entry[0] += 1
        else:
            mask = 0
            for ii in range(5):
                gi_inv = int(inv[best[ii]])
                for jj in range(ii+1, 5):
                    d = int(mul_table[gi_inv, best[jj]])
                    mask |= (1 << orbit_of[d])
            canon_map[best] = [1, mask]

    n_c = len(canon_map)
    masks = np.zeros(n_c, dtype=np.int64)
    mults = np.zeros(n_c, dtype=np.int64)
    for idx, (_, (mult, mask)) in enumerate(canon_map.items()):
        masks[idx] = mask
        mults[idx] = mult

    t2c = [[] for _ in range(n_orbits)]
    for idx, (_, (_, mask)) in enumerate(canon_map.items()):
        for oi in range(n_orbits):
            if mask & (1 << oi):
                t2c[oi].append(idx)
    t2c_np = [np.array(tc, dtype=np.int32) for tc in t2c]

    print(f" {n_c} canonical orbits  ({time.time()-t0:.1f}s)")
    return masks, mults, t2c_np


# ─── Violation counting (numpy) ───────────────────────────────────────────────

def count_v(asn, masks, mults):
    m = masks & asn
    return int(np.sum(mults[(m == 0) | (m == masks)]))


def delta_v(asn, oi, masks, mults, t2c):
    idxs = t2c[oi]
    sm, mu = masks[idxs], mults[idxs]
    new = asn ^ (1 << oi)
    mb = sm & asn;  ma = sm & new
    return int(np.sum(mu * (((ma == 0) | (ma == sm)).astype(np.int64)
                          - ((mb == 0) | (mb == sm)).astype(np.int64))))


# ─── Checkpoint I/O ───────────────────────────────────────────────────────────

def to_flat(asn, mul_table, inv_table, orbit_of):
    """Convert orbit assignment int to 861-edge flat array (Rust format)."""
    flat = []
    for u in range(42):
        u_inv = int(inv_table[u])
        for v in range(u+1, 42):
            d = int(mul_table[u_inv, v])
            flat.append(int((asn >> orbit_of[d]) & 1))
    return flat


def save_checkpoint(path, asn, viol, restart, total_iters, snap_count,
                    mul_table, inv_table, orbit_of):
    flat = to_flat(asn, mul_table, inv_table, orbit_of)
    cp = {
        'n': 42, 's': 5, 'violations': viol,
        'restart': restart, 'total_iters': total_iters,
        'snap_count': snap_count,
        'timestamp': time.strftime('%Y-%m-%d'),
        'assignment': flat,
    }
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(cp))
    tmp.replace(path)
    snap = path.parent / path.name.replace('_checkpoint', f'_snap_{snap_count:03d}_v{viol}')
    snap.write_text(json.dumps(cp))
    print(f"  [saved] {path}  violations={viol}  restart={restart}")


# ─── Tabu search with restarts ────────────────────────────────────────────────

def run(group_name, steps_per_restart, n_restarts, tabu_tenure, seed, out_dir):
    mul, inv, gname = GROUPS[group_name]()
    print(f"\n  Group: {gname}")

    orbits = find_orbits(inv)
    orbit_of = make_orbit_of(orbits)
    n_orbits = len(orbits)
    sizes = [len(o) for o in orbits]
    n_pairs = sizes.count(2)
    n_sing  = sizes.count(1)
    print(f"  Orbit variables: {n_orbits}  ({n_pairs} pairs + {n_sing} singletons)")
    print(f"  Search space: 2^{n_orbits} ~ {2**n_orbits:.2e} colorings")

    masks, mults, t2c = build_lookup(mul, inv, orbit_of, n_orbits)

    ckpt_path = Path(out_dir) / f"RAMSEY_K42_S5_cayley_{group_name}_checkpoint.json"
    rng = random.Random(seed)
    global_best_v = None
    global_best_asn = None
    snap_count = 0
    total_iters = 0
    t_global = time.time()

    for restart in range(n_restarts):
        elapsed = time.time() - t_global
        print(f"\n{'-'*50}")
        print(f"  Restart {restart+1}/{n_restarts}  (elapsed {elapsed:.0f}s, "
              f"total_iters={total_iters})")

        # Initialise: random or perturb from global best
        if global_best_asn is None or restart == 0:
            asn = sum(rng.randint(0, 1) << oi for oi in range(n_orbits))
        else:
            # Flip a random fraction of orbit variables
            frac = 0.10 + 0.05 * min(restart // 5, 4)
            n_flip = max(1, int(n_orbits * frac))
            asn = global_best_asn
            for oi in rng.sample(range(n_orbits), n_flip):
                asn ^= (1 << oi)

        viol = count_v(asn, masks, mults)
        best_asn, best_v = asn, viol
        tabu = [0] * n_orbits  # tabu[oi] = step until released
        no_improve = 0

        print(f"  Starting violations: {viol}")
        t0 = time.time()

        for step in range(steps_per_restart):
            total_iters += 1
            if viol == 0:
                print("  *** VALID COLORING FOUND — R(5,5) ≥ 44 PROVEN ***")
                save_checkpoint(ckpt_path, asn, 0, restart, total_iters,
                                snap_count, mul, inv, orbit_of)
                return

            if step % 2000 == 0:
                rate = (step + 1) / (time.time() - t0 + 1e-9)
                print(f"  iter {step:>7}  viol={viol:>5}  best={best_v:>5}  "
                      f"global={global_best_v or '?':>5}  {rate:>7.0f}/s")

            # Pick best non-tabu flip (with aspiration)
            best_d, best_oi = None, -1
            for oi in range(n_orbits):
                d = delta_v(asn, oi, masks, mults, t2c)
                aspiration = (viol + d < (global_best_v or 10**9))
                if tabu[oi] > step and not aspiration:
                    continue
                if best_d is None or d < best_d:
                    best_d, best_oi = d, oi

            if best_oi == -1:
                best_oi = rng.randrange(n_orbits)
                best_d = delta_v(asn, best_oi, masks, mults, t2c)

            asn ^= (1 << best_oi)
            tabu[best_oi] = step + tabu_tenure
            viol += best_d

            if viol < best_v:
                best_v, best_asn = viol, asn
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= steps_per_restart // 2:
                    print(f"  [plateau] step {step}, best={best_v}")
                    break

        # Update global best
        if global_best_v is None or best_v < global_best_v:
            global_best_v = best_v
            global_best_asn = best_asn
            snap_count += 1
            print(f"\n  *** New global best: {global_best_v} violations ***")
            save_checkpoint(ckpt_path, global_best_asn, global_best_v,
                            restart, total_iters, snap_count, mul, inv, orbit_of)

    elapsed = time.time() - t_global
    print(f"\n{'='*50}")
    print(f"  Done. Best: {global_best_v} violations  "
          f"({n_restarts} restarts, {total_iters} iters, {elapsed:.0f}s)")
    return global_best_v, global_best_asn


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group',    default='d21',   choices=GROUPS.keys())
    ap.add_argument('--steps',    type=int, default=50_000)
    ap.add_argument('--restarts', type=int, default=20)
    ap.add_argument('--tabu',     type=int, default=8)
    ap.add_argument('--seed',     type=int, default=42)
    ap.add_argument('--out-dir',  default='datasets')
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  Cayley Ramsey Search — K_42, avoid K_5")
    print(f"  Group: {args.group}  Steps/restart: {args.steps}  "
          f"Restarts: {args.restarts}  Tabu: {args.tabu}  Seed: {args.seed}")
    print(f"{'='*60}\n")

    run(args.group, args.steps, args.restarts, args.tabu, args.seed, args.out_dir)
