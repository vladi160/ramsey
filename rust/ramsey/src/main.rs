use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "ramsey")]
struct Args {
    #[arg(long, default_value_t = 42)]       n: usize,
    #[arg(long, default_value_t = 5)]        s: usize,
    #[arg(long, default_value_t = 200_000)]  iter: usize,
    #[arg(long, default_value_t = 500)]      restarts: usize,
    #[arg(long, default_value = "datasets")] snap_dir: PathBuf,
    #[arg(long)]                             checkpoint: Option<PathBuf>,
    /// Print per-iteration line every N iters (0 = off)
    #[arg(long, default_value_t = 1000)]     log_every: usize,
    /// Tabu tenure: recently flipped edges are forbidden for this many iters
    #[arg(long, default_value_t = 12)]       tabu: usize,
    /// Stop a restart early if per-restart best hasn't improved in this many iters (0 = disabled)
    #[arg(long, default_value_t = 20_000)]   plateau: usize,
    /// RNG seed (0 = from entropy)
    #[arg(long, default_value_t = 0u64)]     seed: u64,
}

// ── Edge indexing ─────────────────────────────────────────────────────────────

#[inline(always)]
fn eid(u: usize, v: usize, n: usize) -> usize {
    let (a, b) = if u < v { (u, v) } else { (v, u) };
    a * n - a * (a + 1) / 2 + b - a - 1
}

// ── Clique generation ─────────────────────────────────────────────────────────

fn gen_cliques(n: usize, s: usize) -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    fn go(n: usize, s: usize, start: usize, cur: &mut Vec<u16>, out: &mut Vec<Vec<u16>>) {
        if cur.len() == s {
            out.push(cur.clone());
            return;
        }
        let need = s - cur.len();
        for v in start..=(n - need) {
            cur.push(v as u16);
            go(n, s, v + 1, cur, out);
            cur.pop();
        }
    }
    go(n, s, 0, &mut vec![], &mut out);
    out
}

// ── Index (immutable after build) ─────────────────────────────────────────────

struct Index {
    n: usize,
    s: usize,
    num_edges: usize,
    max_sum: u8,
    c2e: Vec<Vec<u32>>,
    e2c: Vec<Vec<u32>>,
}

impl Index {
    fn build(n: usize, s: usize) -> Self {
        println!("  Building clique structure: n={n}, s={s}");
        let t = Instant::now();
        let num_edges = n * (n - 1) / 2;
        let ce_per = s * (s - 1) / 2;
        let max_sum = ce_per as u8;
        let cliques = gen_cliques(n, s);
        let nc = cliques.len();

        let mut c2e: Vec<Vec<u32>> = Vec::with_capacity(nc);
        for cl in &cliques {
            let mut ces = Vec::with_capacity(ce_per);
            for i in 0..s {
                for j in (i + 1)..s {
                    ces.push(eid(cl[i] as usize, cl[j] as usize, n) as u32);
                }
            }
            c2e.push(ces);
        }

        let mut e2c: Vec<Vec<u32>> = vec![vec![]; num_edges];
        for (ci, ces) in c2e.iter().enumerate() {
            for &e in ces {
                e2c[e as usize].push(ci as u32);
            }
        }

        println!("  Total cliques: {nc:>10}  ({:.1}s)", t.elapsed().as_secs_f32());
        Self { n, s, num_edges, max_sum, c2e, e2c }
    }
}

// ── Mutable search state ──────────────────────────────────────────────────────

struct State {
    assignment: Vec<u8>,
    csum: Vec<u8>,
    vcnt: Vec<u32>,
    total: i32,
}

impl State {
    fn from_assignment(idx: &Index, assignment: Vec<u8>) -> Self {
        let nc = idx.c2e.len();
        let mut csum = vec![0u8; nc];
        let mut vcnt = vec![0u32; idx.num_edges];
        let mut total = 0i32;
        for (ci, ces) in idx.c2e.iter().enumerate() {
            let s: u8 = ces.iter().map(|&e| assignment[e as usize]).sum();
            csum[ci] = s;
            if s == 0 || s == idx.max_sum {
                total += 1;
                for &e in ces {
                    vcnt[e as usize] += 1;
                }
            }
        }
        State { assignment, csum, vcnt, total }
    }
}

// ── Core flip operations ──────────────────────────────────────────────────────

#[inline]
fn flip_delta(idx: &Index, st: &State, e: usize) -> i32 {
    let diff: i8 = if st.assignment[e] == 0 { 1 } else { -1 };
    let mut d = 0i32;
    for &ci in &idx.e2c[e] {
        let old = st.csum[ci as usize];
        let new = (old as i8 + diff) as u8;
        d += (new == 0 || new == idx.max_sum) as i32
           - (old == 0 || old == idx.max_sum) as i32;
    }
    d
}

fn apply_flip(idx: &Index, st: &mut State, e: usize) {
    let diff: i8 = if st.assignment[e] == 0 { 1 } else { -1 };
    st.assignment[e] ^= 1;
    for i in 0..idx.e2c[e].len() {
        let ci = idx.e2c[e][i] as usize;
        let old = st.csum[ci];
        let new_s = (old as i8 + diff) as u8;
        st.csum[ci] = new_s;
        let ow = old == 0 || old == idx.max_sum;
        let nw = new_s == 0 || new_s == idx.max_sum;
        if ow != nw {
            let dv: i32 = if nw { 1 } else { -1 };
            st.total += dv;
            for j in 0..idx.c2e[ci].len() {
                let e2 = idx.c2e[ci][j] as usize;
                st.vcnt[e2] = (st.vcnt[e2] as i32 + dv) as u32;
            }
        }
    }
}

// ── Seeds ─────────────────────────────────────────────────────────────────────

fn paley_seed(idx: &Index, rng: &mut impl Rng) -> Vec<u8> {
    let n = idx.n;
    let p = 17usize;

    let mut assignment: Vec<u8> = (0..idx.num_edges).map(|_| rng.gen_range(0u8..2)).collect();

    // Paley(17): QR mod 17 -> red(0), non-QR -> blue(1)
    if n >= p {
        let qr: std::collections::HashSet<usize> = (1..p).map(|x| (x * x) % p).collect();
        for u in 0..p {
            for v in (u + 1)..p {
                assignment[eid(u, v, n)] = if qr.contains(&((v - u) % p)) { 0 } else { 1 };
            }
        }
    }

    // One-pass greedy improvement (shuffled edge order)
    let mut state = State::from_assignment(idx, assignment);
    let mut order: Vec<usize> = (0..idx.num_edges).collect();
    order.shuffle(rng);
    for e in order {
        if flip_delta(idx, &state, e) < 0 {
            apply_flip(idx, &mut state, e);
        }
    }
    state.assignment
}

fn perturb(assignment: &[u8], frac: f64, rng: &mut impl Rng) -> Vec<u8> {
    let mut a = assignment.to_vec();
    let n_flip = ((a.len() as f64 * frac).round() as usize).max(5);
    let mut indices: Vec<usize> = (0..a.len()).collect();
    indices.partial_shuffle(rng, n_flip);
    for &e in &indices[..n_flip] {
        a[e] ^= 1;
    }
    a
}

// ── D3 structural diagnostics ─────────────────────────────────────────────────

/// For each edge (a,b) of `col`, count triangles of `col` containing it.
/// High count → edge participates in many K4s → primary flip target.
/// O(n^3) but n=42 so <1 ms.
fn triangle_count_per_edge(idx: &Index, assignment: &[u8], col: u8) -> Vec<u32> {
    let n = idx.n;
    let ne = idx.num_edges;
    let mut tri = vec![0u32; ne];
    for a in 0..n {
        for b in (a + 1)..n {
            if assignment[eid(a, b, n)] != col { continue; }
            for c in (b + 1)..n {
                if assignment[eid(a, c, n)] == col && assignment[eid(b, c, n)] == col {
                    tri[eid(a, b, n)] += 1;
                    tri[eid(a, c, n)] += 1;
                    tri[eid(b, c, n)] += 1;
                }
            }
        }
    }
    tri
}

/// D3-targeted perturbation: flip edges preferentially from the hub nucleus
/// (top `nucleus_size` vertices by degree), scoring each edge by its triangle
/// participation in the current color.  2/3 of flips are targeted, 1/3 random.
///
/// Rationale: the hub nucleus concentrates K4 overload (r(deg,K4)=0.84).
/// Flipping their highest-triangle edges breaks the most K4s per flip,
/// disrupting the clique structure while preserving the near-regular degree regime.
fn perturb_targeted(idx: &Index, assignment: &[u8], frac: f64, rng: &mut impl Rng) -> Vec<u8> {
    let n = idx.n;
    let n_flip = ((idx.num_edges as f64 * frac).round() as usize).max(5);

    // Per-vertex degree in each color
    let mut red_deg = vec![0u32; n];
    let mut blue_deg = vec![0u32; n];
    for a in 0..n {
        for b in (a + 1)..n {
            let e = eid(a, b, n);
            if assignment[e] == 1 { red_deg[a] += 1; red_deg[b] += 1; }
            else                  { blue_deg[a] += 1; blue_deg[b] += 1; }
        }
    }

    // Triangle count per edge in each color
    let tri_red  = triangle_count_per_edge(idx, assignment, 1);
    let tri_blue = triangle_count_per_edge(idx, assignment, 0);

    // Hub nucleus: top nucleus_size vertices by total degree (=n-1 always in K_n,
    // so use the color-max degree as the overload proxy)
    let nucleus_size = (n / 7).max(4);
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| {
        red_deg[b].max(blue_deg[b]).cmp(&red_deg[a].max(blue_deg[a]))
    });
    let nucleus: std::collections::HashSet<usize> =
        ranked[..nucleus_size].iter().cloned().collect();

    // Score nucleus-incident edges by triangle count in their color
    let mut cands: Vec<(u32, usize)> = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            if !nucleus.contains(&a) && !nucleus.contains(&b) { continue; }
            let e = eid(a, b, n);
            let tri = if assignment[e] == 1 { tri_red[e] } else { tri_blue[e] };
            if tri > 0 {
                cands.push((tri, e));
            }
        }
    }
    cands.sort_unstable_by(|x, y| y.0.cmp(&x.0));

    let mut result = assignment.to_vec();

    // Surgical: flip exactly N_SURGICAL hub-nucleus edges with highest K4 contribution.
    // This disrupts the clique overload at the hub core without destroying the global structure.
    // The rest of the budget is random perturbation (preserves diversity + degree balance).
    const N_SURGICAL: usize = 5;
    let n_targeted = N_SURGICAL.min(cands.len());
    for &(_, e) in cands.iter().take(n_targeted) {
        result[e] ^= 1;
    }

    // Remaining flips: standard random perturbation
    let n_random = n_flip.saturating_sub(n_targeted);
    if n_random > 0 {
        let mut indices: Vec<usize> = (0..idx.num_edges).collect();
        indices.partial_shuffle(rng, n_random);
        for &e in &indices[..n_random] {
            result[e] ^= 1;
        }
    }

    result
}

// ── Checkpoint ────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    n: usize,
    s: usize,
    violations: i32,
    restart: usize,
    total_iters: u64,
    snap_count: u32,
    timestamp: String,
    assignment: Vec<u8>,
}

fn save_ckpt(path: &Path, snap_dir: &Path, cp: &Checkpoint) {
    let json = serde_json::to_string(cp).expect("serialize checkpoint");
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, &json).expect("write checkpoint tmp");
    fs::rename(&tmp, path).expect("rename checkpoint");
    println!("  [checkpoint] saved -> {}  violations={}  restart={}  total_iters={}",
        path.display(), cp.violations, cp.restart, cp.total_iters);

    let stem = path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .replace("_checkpoint", "");
    let snap_path = snap_dir.join(format!(
        "{stem}_snap_{:03}_v{}.json",
        cp.snap_count, cp.violations
    ));
    fs::write(&snap_path, &json).expect("write snapshot");
    println!("  [snapshot]   saved -> {}", snap_path.display());
}

fn load_ckpt(path: &Path) -> Option<Checkpoint> {
    if !path.exists() {
        return None;
    }
    let data = fs::read_to_string(path).ok()?;
    let cp: Checkpoint = serde_json::from_str(&data).ok()?;
    println!("  [checkpoint] loaded <- {}", path.display());
    println!("    violations={}  restart={}  total_iters={}",
        cp.violations, cp.restart, cp.total_iters);
    Some(cp)
}

// ── Local search ──────────────────────────────────────────────────────────────

struct SearchResult {
    best_assignment: Vec<u8>,
    best_violations: i32,
    iters_done: usize,
}

fn local_search(
    idx: &Index,
    init: Vec<u8>,
    max_iter: usize,
    tabu_tenure: usize,
    plateau_limit: usize,
    log_every: usize,
    global_best: i32,
    t0: Instant,
    rng: &mut impl Rng,
) -> SearchResult {
    let mut st = State::from_assignment(idx, init);
    let mut best_a = st.assignment.clone();
    let mut best_v = st.total;
    let mut scratch: Vec<u32> = Vec::with_capacity(16_000);
    let mut tabu = vec![0usize; idx.num_edges];
    let mut iters_no_improve = 0usize;

    println!("\n  IRDME-guided search: {max_iter} iterations");
    println!("  Starting violations: {}", st.total);

    for it in 0..max_iter {
        if st.total == 0 {
            println!("  [iter {it}] SOLVED! Valid coloring found.");
            return SearchResult { best_assignment: st.assignment, best_violations: 0, iters_done: it };
        }

        if log_every > 0 && it % log_every == 0 {
            let elapsed = t0.elapsed().as_secs_f32();
            println!(
                "  iter {:>6}  violations={:>5}  best={:>5}  global={:>5}  ({:.0}s)",
                it, st.total, best_v, global_best, elapsed
            );
        }

        // Find edge with highest vcnt
        let best_e = (0..idx.num_edges)
            .max_by_key(|&e| st.vcnt[e])
            .unwrap();

        if st.vcnt[best_e] == 0 {
            break;
        }

        // Collect violated cliques containing best_e
        scratch.clear();
        for &ci in &idx.e2c[best_e] {
            let s = st.csum[ci as usize];
            if s == 0 || s == idx.max_sum {
                scratch.push(ci);
            }
        }

        let ci = *scratch.choose(rng).unwrap() as usize;

        // Try all edges in chosen clique. Respect tabu unless the move reaches
        // a new global best (aspiration criterion).
        let best_improving = idx.c2e[ci]
            .iter()
            .map(|&e| (e as usize, flip_delta(idx, &st, e as usize)))
            .filter(|&(e, d)| tabu[e] <= it || st.total + d < best_v) // tabu + aspiration
            .min_by_key(|&(_, d)| d);

        let flip_e = match best_improving {
            Some((e, _)) => e,
            None => {
                // All candidates are tabu — pick least-bad ignoring tabu
                idx.c2e[ci]
                    .iter()
                    .map(|&e| (e as usize, flip_delta(idx, &st, e as usize)))
                    .min_by_key(|&(_, d)| d)
                    .unwrap()
                    .0
            }
        };

        apply_flip(idx, &mut st, flip_e);
        tabu[flip_e] = it + tabu_tenure;

        if st.total < best_v {
            best_v = st.total;
            best_a = st.assignment.clone();
            iters_no_improve = 0;
        } else {
            iters_no_improve += 1;
            if plateau_limit > 0 && iters_no_improve >= plateau_limit {
                if log_every > 0 {
                    println!(
                        "  [plateau] no improvement for {plateau_limit} iters, ending restart (best={best_v})"
                    );
                }
                break;
            }
        }
    }

    SearchResult { best_assignment: best_a, best_violations: best_v, iters_done: max_iter }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let ckpt_path = args.checkpoint.clone().unwrap_or_else(|| {
        args.snap_dir.join(format!(
            "RAMSEY_K{}_S{}_rust_checkpoint.json",
            args.n, args.s
        ))
    });

    println!("\n{}", "=".repeat(68));
    println!("  IRDME Ramsey Search -- K_{}, avoid monochromatic K_{}", args.n, args.s);
    println!("  Restarts: {}   Iters/restart: {}", args.restarts, args.iter);
    println!("  Checkpoint: {}", ckpt_path.display());
    println!("{}\n", "=".repeat(68));

    let idx = Index::build(args.n, args.s);

    // Load checkpoint
    let (mut best_a, mut best_v, mut start_restart, mut total_iters, mut snap_count) =
        match load_ckpt(&ckpt_path) {
            Some(cp) if cp.n == args.n && cp.s == args.s => {
                let sr = cp.restart + 1;
                println!(
                    "  Resuming from restart {}/{}, best so far: {} violations\n",
                    sr + 1, args.restarts, cp.violations
                );
                (cp.assignment, cp.violations, sr, cp.total_iters, cp.snap_count)
            }
            Some(cp) => {
                println!(
                    "  Checkpoint is for K_{} s={}, ignoring (wrong graph).\n",
                    cp.n, cp.s
                );
                (vec![], i32::MAX, 0, 0u64, 0u32)
            }
            None => (vec![], i32::MAX, 0, 0u64, 0u32),
        };

    let mut rng = if args.seed == 0 {
        SmallRng::from_entropy()
    } else {
        SmallRng::seed_from_u64(args.seed)
    };
    let mut restart_of_last_best = start_restart;
    let t0 = Instant::now();

    // Total restarts is args.restarts; loop from start_restart to args.restarts
    // (matching Python: range(start_restart, args.restarts))
    for restart in start_restart..args.restarts {
        let since_best = restart - restart_of_last_best;
        let elapsed = t0.elapsed().as_secs_f32();

        println!("\n{}", "-".repeat(50));
        println!(
            "  Restart {}/{}  (elapsed {:.0}s, total iters={})",
            restart + 1,
            args.restarts,
            elapsed,
            total_iters
        );

        // Seed selection
        let seed = if restart == start_restart && !best_a.is_empty() {
            let seed_v = State::from_assignment(&idx, best_a.clone()).total;
            println!("  Seed: checkpoint best ({seed_v} violations)");
            best_a.clone()
        } else if !best_a.is_empty() && since_best < 6 {
            let frac = [0.02f64, 0.04, 0.06][since_best % 3];
            let n_flip = ((idx.num_edges as f64 * frac).round() as usize).max(5);
            let a = perturb_targeted(&idx, &best_a, frac, &mut rng);
            let seed_v = State::from_assignment(&idx, a.clone()).total;
            println!(
                "  Seed: D3-targeted ({:.0}%, {n_flip} flips, {}/{} exploit  [{seed_v} violations])",
                frac * 100.0,
                since_best + 1,
                6
            );
            a
        } else if !best_a.is_empty() && since_best % 4 != 0 {
            let intensity = (since_best / 10).min(2);
            let frac = [0.10f64, 0.20, 0.30][intensity];
            let a = perturb_targeted(&idx, &best_a, frac, &mut rng);
            let seed_v = State::from_assignment(&idx, a.clone()).total;
            println!(
                "  Seed: D3-targeted ({:.0}%, stagnated {since_best} restarts  [{seed_v} violations])",
                frac * 100.0
            );
            a
        } else {
            let a = paley_seed(&idx, &mut rng);
            let seed_v = State::from_assignment(&idx, a.clone()).total;
            println!("  Seed: Paley+FPL fresh [{seed_v} violations]");
            a
        };

        let res = local_search(&idx, seed, args.iter, args.tabu, args.plateau, args.log_every, best_v, t0, &mut rng);
        total_iters += res.iters_done as u64;

        if res.best_violations < best_v {
            best_v = res.best_violations;
            best_a = res.best_assignment.clone();
            restart_of_last_best = restart;
            snap_count += 1;

            println!("\n  *** New global best: {best_v} violations ***");

            let cp = Checkpoint {
                n: args.n,
                s: args.s,
                violations: best_v,
                restart,
                total_iters,
                snap_count,
                timestamp: format!("{:.1}s", t0.elapsed().as_secs_f32()),
                assignment: best_a.clone(),
            };
            save_ckpt(&ckpt_path, &args.snap_dir, &cp);
        }

        if best_v == 0 {
            println!("\n{}", "=".repeat(60));
            println!("  SOLVED: valid K_{} coloring found! R({},{}) > {}",
                args.n, args.s, args.s, args.n);
            println!("  Total time: {:.1}s, iters: {total_iters}",
                t0.elapsed().as_secs_f32());
            let sol = args.snap_dir.join(format!(
                "RAMSEY_K{}_S{}_rust_solution.json", args.n, args.s
            ));
            let cp = Checkpoint {
                n: args.n, s: args.s, violations: 0,
                restart, total_iters, snap_count,
                timestamp: format!("{:.1}s", t0.elapsed().as_secs_f32()),
                assignment: best_a.clone(),
            };
            fs::write(&sol, serde_json::to_string(&cp).unwrap()).unwrap();
            println!("  Saved -> {}", sol.display());
            break;
        }
    }

    println!("\n{}", "=".repeat(68));
    println!("  Search complete. Best violations: {best_v}");
    println!("  Total time: {:.1}s, total iters: {total_iters}",
        t0.elapsed().as_secs_f32());
}
