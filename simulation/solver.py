"""
eBOT heuristic vs. exact P2 reference (SCIP).

Produces the summary table

    | N | sigma | eBOT objective | Reference objective | eBOT gap |

where each objective is the mean over `trials` random bandwidth draws and
the gap is the mean of (eBOT - reference) / reference * 100 per trial.

Usage:
    python solver.py                       # default grid, 20 trials
    python solver.py --trials 5 --time-limit 60
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum


# ============================================================
# Parameters
# ============================================================

G = 1000.0          # Gradient size (Mbit)
TAU = 0.1           # Computation latency per depth level (ms)

SEED = 42

# Output folder, relative to this file
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)

# (N, sigma) configurations reported in the results table
CONFIGS = [
    (10, 0.00),
    (10, 0.10),
    (20, 0.00),
    (20, 0.10),
    (30, 0.00),
]


# ============================================================
# Generate bandwidths
# ============================================================

def generate_bandwidths(N, sigma, rng):
    """
    B_i = 100 * q_i Gbps

    q_i ~ Normal(0.5, sigma), clipped to [0.1, 0.9].
    """

    if sigma == 0:
        q = np.full(N, 0.5)
    else:
        q = rng.normal(0.5, sigma, N)
        q = np.clip(q, 0.1, 0.9)

    return 100.0 * q


# ============================================================
# Evaluate a tree
# ============================================================

def evaluate_tree(B, edges, depths):
    """
    Score a tree given as (child, parent) edges.

    T = G / min_i (B_i / (deg_i + 1)) + TAU * max_i depth_i
    """

    N = len(B)

    degrees = np.zeros(N, dtype=int)

    for child, parent in edges:
        degrees[child] += 1
        degrees[parent] += 1

    beta = np.min(B / (degrees + 1))
    D = int(np.max(depths))

    communication_time = G / beta
    depth_time = TAU * D

    return {
        "objective": communication_time + depth_time,
        "beta": beta,
        "depth": D,
        "communication_time": communication_time,
        "depth_time": depth_time,
        "degrees": degrees,
    }


# ============================================================
# eBOT heuristic
# ============================================================

def eBOT_with_root(B, root):
    """
    Greedy eBOT construction with a fixed root.
    The root is not necessarily the highest-bandwidth node.
    """

    N = len(B)

    V = [root]
    E = []

    degrees = np.zeros(N, dtype=int)
    depths = np.full(N, -1, dtype=int)
    depths[root] = 0

    # Add nodes in descending bandwidth order,
    # excluding the fixed root.
    remaining = sorted(
        [i for i in range(N) if i != root],
        key=lambda i: B[i],
        reverse=True
    )

    for i in remaining:

        best_parent = None
        best_cost = float("inf")

        for j in V:

            # Temporary update: attach i under j
            temp_degrees = degrees.copy()
            temp_degrees[j] += 1
            temp_degrees[i] = 1

            beta = np.min(B / (temp_degrees + 1))
            D = max(depths.max(), depths[j] + 1)

            cost = G / beta + TAU * D

            if cost < best_cost:
                best_cost = cost
                best_parent = j

        # Permanently add node
        E.append((i, best_parent))
        V.append(i)

        degrees[best_parent] += 1
        degrees[i] = 1
        depths[i] = depths[best_parent] + 1

    result = evaluate_tree(B, E, depths)
    result.update(edges=E, depths=depths, root=root)

    return result


def eBOT(B):
    """
    Fair eBOT:
    try every possible root and keep the best greedy solution.
    """

    start_time = time.time()

    best = min(
        (eBOT_with_root(B, root) for root in range(len(B))),
        key=lambda s: s["objective"]
    )

    best["runtime"] = time.time() - start_time

    return best


# ============================================================
# Exact P2 solver using SCIP
# ============================================================

def solve_p2_scip(
    B,
    time_limit=300,
    gap_limit=1e-6,
    warm_start=None,
    verbose=False
):
    N = len(B)

    model = Model("P2")

    if not verbose:
        model.hideOutput()

    # ------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------

    # x[i,j] = 1 if i sends its aggregated result to parent j
    x = {
        (i, j): model.addVar(vtype="B", name=f"x_{i}_{j}")
        for i in range(N)
        for j in range(N)
        if i != j
    }

    # z[i] = 1 if i is the root
    z = {
        i: model.addVar(vtype="B", name=f"root_{i}")
        for i in range(N)
    }

    # beta = minimum normalized bandwidth
    beta = model.addVar(
        lb=0.0, ub=float(np.max(B)), vtype="C", name="beta"
    )

    # Node depths
    h = {
        i: model.addVar(lb=0.0, ub=N - 1, vtype="C", name=f"h_{i}")
        for i in range(N)
    }

    # Maximum depth
    D = model.addVar(lb=0.0, ub=N - 1, vtype="C", name="D")

    # Communication time
    T_comm = model.addVar(lb=0.0, vtype="C", name="T_comm")

    # ------------------------------------------------------------
    # Root / parent constraints
    # ------------------------------------------------------------

    # Each node is either the root or has exactly one parent.
    for i in range(N):
        model.addCons(
            quicksum(x[i, j] for j in range(N) if j != i) + z[i] == 1,
            name=f"parent_or_root_{i}"
        )

    # Exactly one root
    model.addCons(
        quicksum(z[i] for i in range(N)) == 1,
        name="one_root"
    )

    # ------------------------------------------------------------
    # Normalized bandwidth: beta * (deg_i + 1) <= B_i
    # ------------------------------------------------------------

    for i in range(N):

        degree = (
            quicksum(x[i, j] for j in range(N) if j != i)
            + quicksum(x[j, i] for j in range(N) if j != i)
        )

        model.addCons(
            beta * (degree + 1) <= B[i],
            name=f"normalized_bandwidth_{i}"
        )

    # ------------------------------------------------------------
    # Depth constraints
    # ------------------------------------------------------------

    M = N

    # Root has depth 0
    for i in range(N):
        model.addCons(
            h[i] <= M * (1 - z[i]),
            name=f"root_depth_{i}"
        )

    # If x[i,j] = 1 (j is the parent of i), then h[i] = h[j] + 1
    for (i, j), x_ij in x.items():

        model.addCons(
            h[i] >= h[j] + 1 - M * (1 - x_ij),
            name=f"depth_lower_{i}_{j}"
        )

        model.addCons(
            h[i] <= h[j] + 1 + M * (1 - x_ij),
            name=f"depth_upper_{i}_{j}"
        )

    # Maximum depth
    for i in range(N):
        model.addCons(h[i] <= D, name=f"max_depth_{i}")

    # ------------------------------------------------------------
    # Communication time: T_comm >= G / beta  <=>  T_comm * beta >= G
    # ------------------------------------------------------------

    model.addCons(T_comm * beta >= G, name="communication_time")

    # ------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------

    model.setObjective(T_comm + TAU * D, "minimize")

    model.setParam("limits/time", time_limit)
    model.setParam("limits/gap", gap_limit)

    # ------------------------------------------------------------
    # Warm start from a known tree (e.g. eBOT), so the reference
    # is never worse than it even when the time limit is hit.
    # ------------------------------------------------------------

    if warm_start is not None:

        sol = model.createSol()
        parent_of = dict(warm_start["edges"])

        for (i, j), x_ij in x.items():
            model.setSolVal(sol, x_ij, 1.0 if parent_of.get(i) == j else 0.0)

        for i in range(N):
            model.setSolVal(sol, z[i], 1.0 if i == warm_start["root"] else 0.0)
            model.setSolVal(sol, h[i], float(warm_start["depths"][i]))

        model.setSolVal(sol, beta, float(warm_start["beta"]))
        model.setSolVal(sol, D, float(warm_start["depth"]))
        model.setSolVal(sol, T_comm, float(warm_start["communication_time"]))

        model.addSol(sol)

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------

    start_time = time.time()
    model.optimize()
    runtime = time.time() - start_time

    # No feasible solution
    if model.getNSols() == 0:
        return None

    # ------------------------------------------------------------
    # Extract solution and re-score it with the shared evaluator
    # ------------------------------------------------------------

    edges = [
        (i, j) for (i, j), x_ij in x.items()
        if model.getVal(x_ij) > 0.5
    ]

    root = next(i for i in range(N) if model.getVal(z[i]) > 0.5)

    depths = np.array([round(model.getVal(h[i])) for i in range(N)])

    result = evaluate_tree(B, edges, depths)
    result.update(
        edges=edges,
        depths=depths,
        root=root,
        runtime=runtime,
        gap=model.getGap(),
        status=str(model.getStatus()),
    )

    return result


# ============================================================
# Benchmark
# ============================================================

def run_benchmark(
    configs=CONFIGS,
    trials=20,
    time_limit=300,
    seed=SEED,
    verbose=True
):
    """
    Run eBOT and SCIP on `trials` random instances per (N, sigma).

    Returns one row per trial.
    """

    rng = np.random.default_rng(seed)

    rows = []

    for N, sigma in configs:

        if verbose:
            print(f"\n=== N = {N}, sigma = {sigma:.2f} ===")

        # With sigma = 0 every draw is identical, so one trial suffices.
        n_trials = 1 if sigma == 0 else trials

        for trial in range(n_trials):

            B = generate_bandwidths(N, sigma, rng)

            heuristic = eBOT(B)
            reference = solve_p2_scip(
                B, time_limit=time_limit, warm_start=heuristic
            )

            if reference is None:
                if verbose:
                    print(f"Trial {trial:3d}: SCIP found no solution")
                continue

            gap = (
                (heuristic["objective"] - reference["objective"])
                / reference["objective"]
                * 100
            )

            rows.append({
                "N": N,
                "sigma": sigma,
                "trial": trial,
                "ebot_objective": heuristic["objective"],
                "ref_objective": reference["objective"],
                "ebot_gap_percent": gap,
                "ebot_beta": heuristic["beta"],
                "ref_beta": reference["beta"],
                "ebot_depth": heuristic["depth"],
                "ref_depth": reference["depth"],
                "ebot_runtime": heuristic["runtime"],
                "ref_runtime": reference["runtime"],
                "ref_mip_gap": reference["gap"],
                "ref_status": reference["status"],
            })

            if verbose:
                print(
                    f"Trial {trial:3d}: "
                    f"eBOT={heuristic['objective']:.3f}, "
                    f"ref={reference['objective']:.3f} "
                    f"({reference['status']}, "
                    f"{reference['runtime']:.1f}s), "
                    f"gap={gap:.2f}%"
                )

    return pd.DataFrame(rows)


def summarize(df):
    """
    Aggregate per-trial rows into the results table.
    """

    summary = (
        df
        .groupby(["N", "sigma"], sort=False)
        .agg(
            ebot_objective=("ebot_objective", "mean"),
            ref_objective=("ref_objective", "mean"),
            ebot_gap_percent=("ebot_gap_percent", "mean"),
            ebot_runtime=("ebot_runtime", "mean"),
            ref_runtime=("ref_runtime", "mean"),
            trials=("trial", "count"),
            ref_optimal=(
                "ref_status",
                lambda s: s.isin(["optimal", "gaplimit"]).sum()
            ),
        )
        .reset_index()
    )

    return summary


def to_markdown(summary):
    """
    Render the summary in the paper's table format.
    """

    lines = [
        "| \\(N\\) | \\(\\sigma\\) | eBOT objective "
        "| Reference objective | eBOT gap "
        "| eBOT Time (ms) | Solver Time (s) |",
        "| ----: | ---------: | -------------: "
        "| ------------------: | -------: "
        "| -------------: | --------------: |",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.N:5d} "
            f"| {row.sigma:10.2f} "
            f"| {row.ebot_objective:14.3f} "
            f"| {row.ref_objective:19.3f} "
            f"| {row.ebot_gap_percent:7.2f}% "
            f"| {row.ebot_runtime * 1000:14.2f} "
            f"| {row.ref_runtime:15.2f} |"
        )

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    df = run_benchmark(
        trials=args.trials,
        time_limit=args.time_limit,
        seed=args.seed,
    )

    summary = summarize(df)

    os.makedirs(args.results_dir, exist_ok=True)

    csv_paths = [
        os.path.join(args.results_dir, "ebot_vs_scip.csv"),
        os.path.join(args.results_dir, "ebot_vs_scip_summary.csv"),
    ]

    df.to_csv(csv_paths[0], index=False)
    summary.to_csv(csv_paths[1], index=False)

    print("\n" + to_markdown(summary))

    not_optimal = summary[summary["ref_optimal"] < summary["trials"]]
    if not not_optimal.empty:
        print(
            "\nNote: SCIP hit the time limit on some instances; "
            "their reference is the best incumbent, not a proven optimum:"
        )
        print(not_optimal[["N", "sigma", "trials", "ref_optimal"]]
              .to_string(index=False))

    print("\nSaved:")
    for path in csv_paths:
        print(f"  {path}")
