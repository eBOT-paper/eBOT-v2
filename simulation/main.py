"""
Aggregation-time simulation: eBOT vs. PS, Ring All-Reduce and ALEPH.

For every sigma, draws NUM_TRIALS random bandwidth vectors, builds each
topology, and scores it with

    T = G / min_i (B_i / (deg_i + 1)) + tau * D

Usage:
    python main.py                 # run, save CSVs in results/ and PNGs in images/
    python main.py --show          # also open the plots
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Experimental settings
# ============================================================

N = 100
B_SCALE = 100.0       # Gbps
B_MEAN = 0.5
B_MIN = 0.1
B_MAX = 0.9

# Gradient size
G = 1000.0            # Mbit

# Computation latency per aggregation level
tau = 0.1             # ms

# Standard deviations to evaluate
SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20]

# Number of independent experiments for each sigma
NUM_TRIALS = 100

# Scalability sweep: network sizes and the sigmas plotted for each
NODE_COUNTS = [10, 25, 50, 100, 200]
SWEEP_SIGMAS = [0.0, 0.10, 0.20]

RANDOM_SEED = 42

# Output folders, relative to this file
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
IMAGES_DIR = os.path.join(HERE, "images")


# ============================================================
# Bandwidth generation
# ============================================================

def generate_bandwidths(sigma, rng, n=N):
    """
    Generate n node bandwidths.

    beta ~ Normal(0.5, sigma), truncated to [0.1, 0.9] by rejection.
    Bandwidth B_i = beta_i * 100 Gbps.

    When sigma = 0, all nodes have exactly 50 Gbps.
    """

    if sigma == 0:
        return np.full(n, B_MEAN) * B_SCALE

    beta = []

    while len(beta) < n:
        samples = rng.normal(B_MEAN, sigma, size=n * 2)
        samples = samples[(samples >= B_MIN) & (samples <= B_MAX)]
        beta.extend(samples.tolist())

    return np.array(beta[:n]) * B_SCALE


# ============================================================
# Objective
# ============================================================

def communication_time(beta):
    """
    G [Mbit] / beta [Gbit/s] = milliseconds.
    """

    return G / beta


def calculate_time(B, degrees, depths):
    """
    T = G / min_i b_i + tau * D,  where b_i = B_i / (degree_i + 1).
    """

    beta = np.min(B / (degrees + 1))
    D = np.max(depths)

    comm = communication_time(beta)
    comp = tau * D

    return {
        "beta": beta,
        "depth": D,
        "communication_time": comm,
        "computation_time": comp,
        "total_time": comm + comp,
    }


# ============================================================
# Topologies
#
# Each builder takes the bandwidth vector B and returns
# (degrees, depths) as integer arrays of length len(B).
# ============================================================

def build_ebot(B):
    """
    Depth-aware eBOT.

    Nodes are processed in descending bandwidth order.
    The node with the largest bandwidth becomes the root.

    For every new node i, every existing node j is evaluated
    as a candidate parent using

        G / beta' + tau * D'

    where beta' is the minimum normalized bandwidth after
    temporarily adding edge (i, j).
    """

    n = len(B)
    order = np.argsort(-B)

    degrees = np.zeros(n, dtype=int)
    depths = np.zeros(n, dtype=int)

    selected = [order[0]]

    for i in order[1:]:

        best_parent = None
        best_cost = np.inf

        for j in selected:

            # Temporarily add edge i -> j
            temp_degrees = degrees.copy()
            temp_degrees[i] += 1
            temp_degrees[j] += 1

            beta = np.min(B / (temp_degrees + 1))
            D = max(depths.max(), depths[j] + 1)

            cost = communication_time(beta) + tau * D

            if cost < best_cost:
                best_cost = cost
                best_parent = j

        # Permanently add the selected edge
        degrees[i] += 1
        degrees[best_parent] += 1
        depths[i] = depths[best_parent] + 1

        selected.append(i)

    return degrees, depths


def build_ps(B):
    """
    Two-layer Parameter Server architecture.

    The highest-bandwidth node is the parameter server;
    all other nodes connect directly to it.
    """

    n = len(B)
    root = np.argmax(B)

    degrees = np.ones(n, dtype=int)
    depths = np.ones(n, dtype=int)

    degrees[root] = n - 1
    depths[root] = 0

    return degrees, depths


def build_ring(B):
    """
    Ring All-Reduce.

    Every node has two incident links. The ring has no hierarchical
    aggregation, so its depth is modelled as the N - 1 sequential
    communication steps of the classical ring algorithm.
    """

    n = len(B)

    degrees = np.full(n, 2, dtype=int)
    depths = np.full(n, n - 1, dtype=int)

    return degrees, depths


def build_aleph(B):
    """
    Simplified three-layer hierarchical parameter-server topology.

    Layer 0: root / parameter server
    Layer 1: sqrt(N) aggregators (highest bandwidth after the root)
    Layer 2: workers, assigned round-robin to aggregators
    """

    n = len(B)
    order = np.argsort(-B)

    root = order[0]
    num_aggregators = max(1, int(np.sqrt(n)))
    aggregators = order[1:num_aggregators + 1]
    workers = order[num_aggregators + 1:]

    degrees = np.zeros(n, dtype=int)
    depths = np.zeros(n, dtype=int)

    # Root -> aggregators
    degrees[root] += len(aggregators)
    degrees[aggregators] += 1
    depths[aggregators] = 1

    # Workers -> aggregators, in node-index order
    for idx, worker in enumerate(np.sort(workers)):
        degrees[worker] += 1
        degrees[aggregators[idx % len(aggregators)]] += 1
        depths[worker] = 2

    return degrees, depths


ARCHITECTURES = {
    "eBOT": build_ebot,
    "PS": build_ps,
    "Ring": build_ring,
    "ALEPH": build_aleph,
}


# ============================================================
# Simulation
# ============================================================

def run_trial(sigma, rng, n=N):
    """
    Score every architecture on one random bandwidth draw of n nodes.
    """

    B = generate_bandwidths(sigma, rng, n)

    return [
        {
            "sigma": sigma,
            "architecture": name,
            **calculate_time(B, *build(B)),
        }
        for name, build in ARCHITECTURES.items()
    ]


def run_simulation(sigmas=SIGMAS, trials=NUM_TRIALS, seed=RANDOM_SEED):
    """
    Returns one row per (sigma, trial, architecture).
    """

    rng = np.random.default_rng(seed)

    rows = []

    for sigma in sigmas:
        for trial in range(trials):
            for result in run_trial(sigma, rng):
                result["trial"] = trial
                rows.append(result)

    return pd.DataFrame(rows)


def run_node_sweep(
    node_counts=NODE_COUNTS,
    sigmas=SWEEP_SIGMAS,
    trials=NUM_TRIALS,
    seed=RANDOM_SEED
):
    """
    Returns one row per (N, sigma, trial, architecture).
    """

    rng = np.random.default_rng(seed)

    rows = []

    for n in node_counts:
        for sigma in sigmas:
            for trial in range(trials):
                for result in run_trial(sigma, rng, n):
                    result["N"] = n
                    result["trial"] = trial
                    rows.append(result)

    return pd.DataFrame(rows)


def summarize(df, by=("sigma",)):

    return (
        df
        .groupby([*by, "architecture"])
        .agg(
            total_time_mean=("total_time", "mean"),
            total_time_std=("total_time", "std"),
            communication_time_mean=("communication_time", "mean"),
            computation_time_mean=("computation_time", "mean"),
            beta_mean=("beta", "mean"),
            beta_std=("beta", "std"),
            depth_mean=("depth", "mean"),
        )
        .reset_index()
    )


# ============================================================
# Plots
# ============================================================

# Fixed identity per architecture: validated categorical slots 1-4,
# plus a distinct marker so series never rely on color alone.
STYLE = {
    "eBOT":  {"color": "#2a78d6", "marker": "o"},
    "PS":    {"color": "#eb6834", "marker": "s"},
    "Ring":  {"color": "#1baf7a", "marker": "^"},
    "ALEPH": {"color": "#eda100", "marker": "D"},
}

# Paper figures: file name -> (column, std column, y label, log scale)
FIGURES = {
    "aggregation_time_vs_sigma": (
        "total_time_mean", "total_time_std",
        "Aggregation time (ms)", True,
    ),
    "effective_bandwidth_vs_sigma": (
        "beta_mean", "beta_std",
        "Min. effective bandwidth (Gbps)", True,
    ),
    "communication_time_vs_sigma": (
        "communication_time_mean", None,
        "Communication time (ms)", True,
    ),
    "depth_vs_sigma": (
        "depth_mean", None,
        "Maximum depth", True,
    ),
}


def plot_metric(
    summary, column, ylabel, yerr_column=None, log=False,
    x="sigma", xlabel=r"Bandwidth heterogeneity $\sigma$", logx=False
):
    """
    One line per architecture of `column` against `x`.
    Sized for a half-width (0.48 textwidth) subfigure.
    """

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for architecture in ARCHITECTURES:

        data = summary[summary["architecture"] == architecture]

        ax.errorbar(
            data[x],
            data[column],
            yerr=None if yerr_column is None else data[yerr_column],
            label=architecture,
            linewidth=2,
            markersize=5,
            capsize=3,
            elinewidth=1,
            **STYLE[architecture],
        )

    if log:
        ax.set_yscale("log")

    if logx:
        ax.set_xscale("log")
        ax.minorticks_off()

    ticks = sorted(summary[x].unique())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ticks)
    if logx:
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.grid(True, which="major", color="#e4e4e2", linewidth=0.6)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(frameon=False, ncol=2, fontsize=7.5)
    fig.tight_layout()

    return fig


def save_figures(summary, outdir=IMAGES_DIR, show=False):
    """
    Save every figure in FIGURES as <outdir>/<name>.png.
    """

    os.makedirs(outdir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
    })

    paths = []

    for name, (column, yerr_column, ylabel, log) in FIGURES.items():

        fig = plot_metric(summary, column, ylabel, yerr_column, log)

        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)

        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return paths


def save_node_figures(sweep_summary, outdir=IMAGES_DIR, show=False):
    """
    Save aggregation time vs. N, one figure per sigma, as
    <outdir>/aggregation_time_vs_nodes_sigma<100*sigma:03d>.png.
    """

    os.makedirs(outdir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
    })

    paths = []

    for sigma in sorted(sweep_summary["sigma"].unique()):

        data = sweep_summary[sweep_summary["sigma"] == sigma]

        fig = plot_metric(
            data, "total_time_mean", "Aggregation time (ms)",
            "total_time_std", log=True,
            x="N", xlabel="Number of nodes $N$", logx=True,
        )

        name = f"aggregation_time_vs_nodes_sigma{round(sigma * 100):03d}"
        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)

        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return paths


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df = run_simulation(trials=args.trials, seed=args.seed)
    summary = summarize(df)

    print("\n===== Simulation Results =====\n")
    print(summary.to_string(index=False))

    os.makedirs(args.results_dir, exist_ok=True)

    csv_paths = [
        os.path.join(args.results_dir, "aggregation_simulation_results.csv"),
        os.path.join(args.results_dir, "aggregation_simulation_summary.csv"),
    ]

    df.to_csv(csv_paths[0], index=False)
    summary.to_csv(csv_paths[1], index=False)

    image_paths = save_figures(summary, args.images_dir, show=args.show)

    # --------------------------------------------------------
    # Scalability: aggregation time vs. number of nodes
    # --------------------------------------------------------

    sweep = run_node_sweep(trials=args.trials, seed=args.seed)
    sweep_summary = summarize(sweep, by=("N", "sigma"))

    print("\n===== Scalability Results =====\n")
    print(sweep_summary.to_string(index=False))

    csv_paths += [
        os.path.join(args.results_dir, "node_sweep_results.csv"),
        os.path.join(args.results_dir, "node_sweep_summary.csv"),
    ]

    sweep.to_csv(csv_paths[2], index=False)
    sweep_summary.to_csv(csv_paths[3], index=False)

    image_paths += save_node_figures(
        sweep_summary, args.images_dir, show=args.show
    )

    print("\nResults saved to:")
    for path in csv_paths + image_paths:
        print(f"  {path}")
