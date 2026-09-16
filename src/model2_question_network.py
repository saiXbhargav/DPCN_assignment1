"""
model2_question_network.py
Model 2: Question-Question Concept Thematic Network.
Analyzes cross-question correlations, identifying core anchor questions,
domain modularity, and inter-thematic bridge concepts.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import THEME_COLORS


def compute_question_correlation(df_numeric):
    """
    Computes pairwise Pearson correlation across all 60 questions.
    """
    return df_numeric.corr(method="pearson")


def build_question_network(df_corr, df_metadata, threshold=0.35):
    """
    Constructs a graph where nodes are questions and edges exist if correlation >= threshold.
    """
    G = nx.Graph()
    meta_dict = df_metadata.set_index("code").to_dict(orient="index")

    for code in df_corr.columns:
        cat = meta_dict.get(code, {}).get("category", "General")
        text = meta_dict.get(code, {}).get("text", "")
        G.add_node(code, category=cat, text=text)

    cols = df_corr.columns.tolist()
    N = len(cols)
    vals = df_corr.values

    for i in range(N):
        for j in range(i + 1, N):
            w = float(vals[i, j])
            if w >= threshold:
                G.add_edge(cols[i], cols[j], weight=w)

    return G


def compute_question_metrics(G, df_numeric, df_metadata):
    """
    Calculates question-level network metrics and consensus/polarization measures.
    """
    meta_dict = df_metadata.set_index("code").to_dict(orient="index")

    deg = dict(G.degree())
    weighted_deg = dict(G.degree(weight="weight"))
    bet = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G)

    # Calculate mean response and variance (polarization)
    means = df_numeric.mean()
    variances = df_numeric.var()

    records = []
    for code in df_numeric.columns:
        meta = meta_dict.get(code, {})
        records.append({
            "code": code,
            "category": meta.get("category", "General"),
            "mean_score": round(float(means.get(code, 0)), 3),
            "variance_polarization": round(float(variances.get(code, 0)), 3),
            "degree": deg.get(code, 0),
            "weighted_strength": round(weighted_deg.get(code, 0.0), 3),
            "betweenness_centrality": round(bet.get(code, 0.0), 4),
            "closeness_centrality": round(closeness.get(code, 0.0), 4),
            "clustering_coefficient": round(clustering.get(code, 0.0), 4),
            "question_text": meta.get("text", ""),
        })

    df_q_metrics = pd.DataFrame(records).sort_values(by="betweenness_centrality", ascending=False)
    return df_q_metrics


def plot_question_correlation_heatmap(df_corr, df_metadata, output_path):
    """
    Plots a 60x60 correlation heatmap with category block boundaries.
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    # Sort columns by theme: T, E, S, V
    sorted_codes = df_metadata.sort_values(by=["category_key", "code"])["code"].tolist()
    sorted_corr = df_corr.loc[sorted_codes, sorted_codes]

    # Clean diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        sorted_corr,
        cmap=cmap,
        vmin=-0.4,
        vmax=0.8,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        ax=ax,
        xticklabels=True,
        yticklabels=True,
    )

    ax.tick_params(axis="both", labelsize=6.5)

    # Draw separator lines between the 4 themes (every 15 questions)
    for boundary in [15, 30, 45]:
        ax.axhline(boundary, color="white", linewidth=2.0)
        ax.axvline(boundary, color="white", linewidth=2.0)

    # Add domain labels
    theme_positions = [7.5, 22.5, 37.5, 52.5]
    theme_labels = ["Technology (T)", "Education (E)", "Society (S)", "Environment (V)"]
    for pos, label in zip(theme_positions, theme_labels):
        ax.text(pos, -1.5, label, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2c3e50")

    ax.set_title("Question-Question Correlation Matrix (60x60 Thematic Blocks)", fontweight="bold", pad=25)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_question_network(G, df_q_metrics, output_path):
    """
    Plots the Question Concept Network colored by Domain, with node size
    scaled by Betweenness Centrality to highlight bridge questions.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, k=0.45, iterations=90, seed=42)

    # Color nodes by thematic category
    color_map = THEME_COLORS
    node_colors = [color_map.get(G.nodes[n].get("category", ""), "#95a5a6") for n in G.nodes()]

    # Size scaled by betweenness centrality
    q_dict = df_q_metrics.set_index("code")["betweenness_centrality"].to_dict()
    node_sizes = [200 + 4000 * q_dict.get(n, 0) for n in G.nodes()]

    # Draw edges with opacity based on weight
    weights = [G[u][v].get("weight", 0.3) for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    edge_alphas = [min(0.9, max(0.2, (w / max_w) * 0.7)) for w in weights]
    edge_widths = [0.6 + 2.0 * (w / max_w) for w in weights]

    for (u, v), w, a in zip(G.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], width=w, alpha=a,
            edge_color="#95a5a6", ax=ax
        )

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        edgecolors="black", linewidths=0.8, ax=ax
    )

    # Label top bridge questions with their code
    top_bridges = df_q_metrics.head(12)["code"].tolist()
    labels = {n: str(n) for n in G.nodes() if n in top_bridges or G.degree(n) > 8}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", font_color="#1a1a1a", ax=ax)

    # Legend for themes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=theme,
               markerfacecolor=color, markersize=10)
        for theme, color in color_map.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Thematic Domains", frameon=True, framealpha=0.95)

    ax.set_title("Question Concept Network\n(Node size ~ Betweenness Centrality; Labels show key thematic bridge questions)",
                 fontweight="bold", pad=15)
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_model2_pipeline(df_numeric, df_metadata, output_dir, threshold=0.35):
    """
    Executes the full end-to-end pipeline for Model 2 (Question Concept Network).
    """
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("\n--- Running Model 2: Question-Question Concept Network ---")

    # 1. Correlation Matrix
    df_corr = compute_question_correlation(df_numeric)

    # 2. Build Question Network
    print(f"Constructing question concept network at threshold tau={threshold}...")
    G_q = build_question_network(df_corr, df_metadata, threshold=threshold)

    # 3. Metrics Computation
    df_q_metrics = compute_question_metrics(G_q, df_numeric, df_metadata)
    df_q_metrics.to_csv(os.path.join(tables_dir, "question_centralities.csv"), index=False)

    # 4. Generate Visualizations
    print("Generating Model 2 figures...")
    plot_question_correlation_heatmap(df_corr, df_metadata, os.path.join(figures_dir, "fig5_question_correlation_heatmap.png"))
    plot_question_network(G_q, df_q_metrics, os.path.join(figures_dir, "fig6_question_thematic_network.png"))

    print("Model 2 execution finished successfully.")
    return G_q, df_corr, df_q_metrics
