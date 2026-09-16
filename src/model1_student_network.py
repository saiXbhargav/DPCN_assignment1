"""
model1_student_network.py
Model 1: Student-Student Opinion Alignment Network.
Computes pairwise similarities, percolation thresholding, community detection (Louvain),
centralities, and benchmarks against Erdős-Rényi (ER) random graphs.
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Aesthetic plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def compute_student_similarity(df_numeric, metric="pearson"):
    """
    Computes pairwise similarity matrix between students.
    metric: 'pearson' or 'cosine'
    """
    if metric == "pearson":
        # Pearson correlation across the 60 questions
        sim_matrix = df_numeric.T.corr(method="pearson").values
    elif metric == "cosine":
        sim_matrix = cosine_similarity(df_numeric.values)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    student_ids = df_numeric.index.tolist()
    df_sim = pd.DataFrame(sim_matrix, index=student_ids, columns=student_ids)
    return df_sim


def percolation_analysis(df_sim, thresholds=None, output_dir=None):
    """
    Percolation analysis across similarity thresholds tau to choose the optimal cutoff.
    Evaluates:
    - Number of edges
    - Fraction of nodes in the Giant Connected Component (GCC)
    - Number of isolated nodes
    - Average Clustering Coefficient
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.75, 14)

    N = len(df_sim)
    sim_vals = df_sim.values
    results = []

    for tau in thresholds:
        # Binary adjacency: 1 if sim >= tau and i != j
        adj = (sim_vals >= tau).astype(int)
        np.fill_diagonal(adj, 0)
        G = nx.from_numpy_array(adj)

        num_edges = G.number_of_edges()
        density = nx.density(G)
        isolates = len(list(nx.isolates(G)))

        if num_edges > 0:
            gcc_nodes = max(nx.connected_components(G), key=len)
            gcc_fraction = len(gcc_nodes) / N
            avg_clustering = nx.average_clustering(G)
        else:
            gcc_fraction = 0.0
            avg_clustering = 0.0

        results.append({
            "threshold": round(tau, 3),
            "edges": num_edges,
            "density": round(density, 4),
            "gcc_fraction": round(gcc_fraction, 4),
            "isolates": isolates,
            "avg_clustering": round(avg_clustering, 4),
        })

    df_percolation = pd.DataFrame(results)

    # Plot percolation curves
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(8.5, 5))

        color1 = "#1f77b4"
        color2 = "#d95f02"
        color3 = "#2ca02c"

        ax1.set_xlabel(r"Similarity Threshold ($\tau$)", fontweight="bold")
        ax1.set_ylabel("Fraction / Value", color=color1, fontweight="bold")
        l1 = ax1.plot(df_percolation["threshold"], df_percolation["gcc_fraction"],
                      marker="o", color=color1, label="Giant Component Fraction (GCC)", linewidth=2)
        l2 = ax1.plot(df_percolation["threshold"], df_percolation["density"],
                      marker="s", linestyle="--", color=color2, label="Graph Density", linewidth=1.8)
        l3 = ax1.plot(df_percolation["threshold"], df_percolation["avg_clustering"],
                      marker="^", linestyle="-.", color=color3, label="Avg. Clustering Coeff. (C)", linewidth=1.8)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(-0.05, 1.05)

        # Secondary axis for number of isolates
        ax2 = ax1.twinx()
        color_iso = "#7570b3"
        ax2.set_ylabel("Number of Isolated Nodes", color=color_iso, fontweight="bold")
        l4 = ax2.plot(df_percolation["threshold"], df_percolation["isolates"],
                      marker="x", color=color_iso, label="Isolated Nodes", linewidth=1.8, linestyle=":")
        ax2.tick_params(axis="y", labelcolor=color_iso)
        ax2.grid(False)

        # Combine legends
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center left", frameon=True, framealpha=0.9)

        plt.title(r"Percolation Analysis: Network Connectivity vs. Similarity Threshold ($\tau$)",
                  fontweight="bold", pad=12)
        fig.tight_layout()

        plot_path = os.path.join(output_dir, "fig1_percolation_analysis.png")
        plt.savefig(plot_path)
        plt.close()

    return df_percolation


def build_student_network(df_sim, threshold=0.35):
    """
    Builds both unweighted (binary) and weighted NetworkX graphs at a chosen threshold tau.
    Self-loops are removed.
    """
    student_ids = df_sim.index.tolist()
    N = len(student_ids)
    sim_vals = df_sim.values

    G = nx.Graph()
    for sid in student_ids:
        G.add_node(str(sid))

    for i in range(N):
        for j in range(i + 1, N):
            w = float(sim_vals[i, j])
            if w >= threshold:
                G.add_edge(str(student_ids[i]), str(student_ids[j]), weight=w)

    return G


def compute_network_metrics(G):
    """
    Computes node-level centralities and global network statistics.
    """
    # Node centralities
    deg_centrality = nx.degree_centrality(G)
    node_degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G)

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes()}

    df_nodes = pd.DataFrame({
        "student_id": list(G.nodes()),
        "degree": [node_degree[n] for n in G.nodes()],
        "degree_centrality": [round(deg_centrality[n], 4) for n in G.nodes()],
        "weighted_strength": [round(weighted_degree[n], 4) for n in G.nodes()],
        "betweenness_centrality": [round(betweenness[n], 4) for n in G.nodes()],
        "closeness_centrality": [round(closeness[n], 4) for n in G.nodes()],
        "eigenvector_centrality": [round(eigenvector[n], 4) for n in G.nodes()],
        "clustering_coefficient": [round(clustering[n], 4) for n in G.nodes()],
    }).set_index("student_id")

    # Global properties
    N = G.number_of_nodes()
    M = G.number_of_edges()
    density = nx.density(G)
    avg_k = (2 * M) / N if N > 0 else 0
    avg_C = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    # Component metrics
    components = list(nx.connected_components(G))
    gcc_nodes = max(components, key=len) if components else set()
    GCC = G.subgraph(gcc_nodes).copy()

    if len(GCC) > 1 and nx.is_connected(GCC):
        avg_L = nx.average_shortest_path_length(GCC)
        diameter = nx.diameter(GCC)
    else:
        avg_L = float("nan")
        diameter = float("nan")

    assortativity = nx.degree_pearson_correlation_coefficient(G) if M > 1 else float("nan")

    global_metrics = {
        "num_nodes": N,
        "num_edges": M,
        "average_degree": round(avg_k, 3),
        "density": round(density, 4),
        "avg_clustering_coeff": round(avg_C, 4),
        "transitivity": round(transitivity, 4),
        "gcc_nodes": len(gcc_nodes),
        "gcc_fraction": round(len(gcc_nodes) / N, 4),
        "num_components": len(components),
        "avg_path_length_gcc": round(avg_L, 4),
        "diameter_gcc": diameter,
        "degree_assortativity": round(assortativity, 4),
    }

    return df_nodes, global_metrics


def benchmark_erdos_renyi(N, M, empirical_C, empirical_L, num_simulations=50):
    """
    Simulates an ensemble of Erdős-Rényi G(N, p) random graphs to compute null model
    clustering coefficient and path length for Small-World Index calculation.
    """
    if N <= 1 or M == 0:
        return {}

    p = (2.0 * M) / (N * (N - 1))
    c_rand_list = []
    l_rand_list = []

    for seed in range(num_simulations):
        G_er = nx.erdos_renyi_graph(N, p, seed=seed)
        c_rand_list.append(nx.average_clustering(G_er))
        if nx.is_connected(G_er):
            l_rand_list.append(nx.average_shortest_path_length(G_er))
        else:
            comps = list(nx.connected_components(G_er))
            gcc = G_er.subgraph(max(comps, key=len))
            if len(gcc) > 1:
                l_rand_list.append(nx.average_shortest_path_length(gcc))

    c_er_mean = np.mean(c_rand_list)
    l_er_mean = np.mean(l_rand_list)

    # Theoretical ER values
    c_er_theory = p
    avg_k = (2.0 * M) / N
    l_er_theory = np.log(N) / np.log(avg_k) if avg_k > 1 else float("nan")

    # Small-world index S = (C / C_rand) / (L / L_rand)
    gamma = empirical_C / c_er_mean if c_er_mean > 0 else float("nan")
    lambda_val = empirical_L / l_er_mean if l_er_mean > 0 else float("nan")
    small_world_sigma = gamma / lambda_val if lambda_val > 0 else float("nan")

    return {
        "er_probability_p": round(p, 4),
        "er_clustering_simulated_mean": round(c_er_mean, 4),
        "er_clustering_simulated_std": round(float(np.std(c_rand_list)), 4),
        "er_clustering_theory": round(c_er_theory, 4),
        "er_path_length_simulated_mean": round(l_er_mean, 4),
        "er_path_length_simulated_std": round(float(np.std(l_rand_list)), 4),
        "er_path_length_theory": round(l_er_theory, 4),
        "clustering_ratio_gamma": round(gamma, 4),
        "path_length_ratio_lambda": round(lambda_val, 4),
        "small_world_index_sigma": round(small_world_sigma, 4),
    }


def detect_communities_and_profiles(G, df_numeric, df_metadata):
    """
    Detects opinion communities using Louvain modularity maximization.
    Calculates ideological profiles per community across the 4 survey themes.
    """
    # Louvain communities on weighted graph
    communities = list(nx.community.louvain_communities(G, weight="weight", seed=42))
    modularity = nx.community.modularity(G, communities, weight="weight")

    # Sort communities by size descending
    communities = sorted(communities, key=len, reverse=True)

    node_to_comm = {}
    for comm_idx, comm_set in enumerate(communities):
        for node in comm_set:
            node_to_comm[str(node)] = comm_idx + 1

    # Map question categories
    q_to_cat = dict(zip(df_metadata["code"], df_metadata["category"]))

    profiles = []
    for comm_idx, comm_set in enumerate(communities):
        nodes_list = [str(n) for n in comm_set if str(n) in df_numeric.index]
        if not nodes_list:
            continue
        sub_df = df_numeric.loc[nodes_list]

        # Calculate average response for each theme
        theme_scores = {}
        for theme in ["Technology", "Education", "Society & Ethics", "Environment"]:
            theme_cols = [c for c in sub_df.columns if q_to_cat.get(c) == theme]
            theme_scores[theme] = round(float(sub_df[theme_cols].values.mean()), 3)

        profiles.append({
            "community_id": comm_idx + 1,
            "size": len(comm_set),
            "percentage": round((len(comm_set) / len(G.nodes())) * 100, 1),
            "technology_mean": theme_scores["Technology"],
            "education_mean": theme_scores["Education"],
            "society_mean": theme_scores["Society & Ethics"],
            "environment_mean": theme_scores["Environment"],
            "overall_stance": round(float(sub_df.values.mean()), 3),
        })

    df_profiles = pd.DataFrame(profiles)
    return node_to_comm, modularity, df_profiles


def plot_student_network(G, node_to_comm, df_nodes, output_path):
    """
    Plots the student-student network graph with nodes colored by Louvain community,
    node size proportional to Betweenness Centrality, and layout generated by ForceAtlas2 / Spring.
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    pos = nx.spring_layout(G, k=0.35, iterations=80, seed=42)

    # Color palette for communities
    comm_ids = sorted(list(set(node_to_comm.values())))
    palette = sns.color_palette("tab10", len(comm_ids))
    color_map = {cid: palette[i] for i, cid in enumerate(comm_ids)}

    node_colors = [color_map[node_to_comm.get(n, 1)] for n in G.nodes()]

    # Node sizes scaled by betweenness centrality (with a minimum baseline)
    bet = [df_nodes.loc[n, "betweenness_centrality"] if n in df_nodes.index else 0 for n in G.nodes()]
    node_sizes = [150 + 2500 * b for b in bet]

    # Draw edges with opacity based on weight
    weights = [G[u][v].get("weight", 0.3) for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    edge_alphas = [min(1.0, max(0.15, (w / max_w) * 0.8)) for w in weights]
    edge_widths = [0.6 + 1.8 * (w / max_w) for w in weights]

    for (u, v), w, a in zip(G.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], width=w, alpha=a,
            edge_color="#7f8c8d", ax=ax
        )

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        edgecolors="white", linewidths=1.2, ax=ax
    )

    # Label top 10 highest betweenness "bridge" students
    top_bridge_nodes = df_nodes.sort_values(by="betweenness_centrality", ascending=False).head(8).index.tolist()
    labels = {n: str(n) for n in G.nodes() if n in top_bridge_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", font_color="#111111", ax=ax)

    # Custom legend for communities
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=f"Community {cid}",
               markerfacecolor=color_map[cid], markersize=10)
        for cid in comm_ids
    ]
    ax.legend(handles=legend_elements, loc="upper right", title="Louvain Communities", frameon=True, framealpha=0.95)

    ax.set_title("Student Opinion Alignment Network\n(Node size ~ Betweenness Centrality; Colors ~ Opinion Communities)",
                 fontweight="bold", pad=15)
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_degree_distribution(G, output_path):
    """
    Plots the degree distribution histogram and compares it with Poisson distribution.
    """
    degrees = [d for _, d in G.degree()]
    N = G.number_of_nodes()
    avg_k = np.mean(degrees)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    sns.histplot(degrees, bins=15, kde=True, color="#2b5c8f", stat="density", ax=ax, label="Empirical Degree Distribution")

    # Overlay theoretical Poisson distribution (ER null model)
    from scipy.stats import poisson
    k_vals = np.arange(min(degrees), max(degrees) + 1)
    poisson_pmf = poisson.pmf(k_vals, avg_k)
    ax.plot(k_vals, poisson_pmf, color="#e76f51", linestyle="--", linewidth=2.2, label=f"Poisson Fit (ER Null Model, $\\lambda={avg_k:.1f}$)")

    ax.set_xlabel("Degree ($k$)", fontweight="bold")
    ax.set_ylabel("Density / Probability", fontweight="bold")
    ax.set_title("Degree Distribution vs. Erdős-Rényi Poisson Baseline", fontweight="bold", pad=12)
    ax.legend(frameon=True)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_community_radar_or_bars(df_profiles, output_path):
    """
    Plots grouped bar chart of thematic scores per community to reveal ideological personas.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    themes = ["Technology", "Education", "Society & Ethics", "Environment"]
    x = np.arange(len(themes))
    width = 0.8 / len(df_profiles)

    palette = sns.color_palette("tab10", len(df_profiles))

    for idx, row in df_profiles.iterrows():
        cid = int(row["community_id"])
        scores = [
            row["technology_mean"],
            row["education_mean"],
            row["society_mean"],
            row["environment_mean"],
        ]
        offset = (idx - len(df_profiles) / 2) * width + width / 2
        ax.bar(x + offset, scores, width, label=f"Comm {cid} (N={int(row['size'])})",
               color=palette[idx], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(themes, fontweight="bold")
    ax.set_ylabel("Mean Likert Score (Scale: -2 to +2)", fontweight="bold")
    ax.set_ylim(-0.5, 2.0)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Community Opinion Personas across Thematic Domains", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_model1_pipeline(df_numeric, df_metadata, output_dir, threshold=0.35):
    """
    Executes the full end-to-end pipeline for Model 1 (Student Opinion Network).
    """
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("\n--- Running Model 1: Student-Student Opinion Network ---")

    # 1. Similarity computation
    df_sim = compute_student_similarity(df_numeric, metric="pearson")

    # 2. Percolation Analysis
    print("Performing percolation threshold sweep...")
    df_percolation = percolation_analysis(df_sim, output_dir=figures_dir)
    df_percolation.to_csv(os.path.join(tables_dir, "percolation_analysis.csv"), index=False)

    # 3. Build Network at chosen threshold
    print(f"Constructing student network at threshold tau={threshold}...")
    G = build_student_network(df_sim, threshold=threshold)

    # 4. Compute Metrics
    df_nodes, global_metrics = compute_network_metrics(G)

    # 5. ER Benchmark
    print("Benchmarking with Erdos-Renyi random graphs...")
    er_metrics = benchmark_erdos_renyi(
        N=global_metrics["num_nodes"],
        M=global_metrics["num_edges"],
        empirical_C=global_metrics["avg_clustering_coeff"],
        empirical_L=global_metrics["avg_path_length_gcc"],
        num_simulations=50
    )

    # Combine global and ER metrics into single table
    all_global = {**global_metrics, **er_metrics}
    df_global = pd.DataFrame([all_global])
    df_global.to_csv(os.path.join(tables_dir, "global_network_metrics.csv"), index=False)

    # 6. Community Detection
    print("Detecting communities with Louvain modularity...")
    node_to_comm, modularity, df_profiles = detect_communities_and_profiles(G, df_numeric, df_metadata)
    df_nodes["community"] = [node_to_comm.get(n, 0) for n in df_nodes.index]
    df_nodes.to_csv(os.path.join(tables_dir, "student_centralities.csv"))
    df_profiles.to_csv(os.path.join(tables_dir, "community_profiles.csv"), index=False)

    # 7. Generate Visualizations
    print("Generating Model 1 figures...")
    plot_student_network(G, node_to_comm, df_nodes, os.path.join(figures_dir, "fig2_student_network_communities.png"))
    plot_degree_distribution(G, os.path.join(figures_dir, "fig3_degree_distribution.png"))
    plot_community_radar_or_bars(df_profiles, os.path.join(figures_dir, "fig4_community_opinion_radar.png"))

    print("Model 1 execution finished successfully.")
    return G, df_nodes, all_global, df_profiles
