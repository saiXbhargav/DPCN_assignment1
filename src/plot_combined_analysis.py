"""
Script to generate publication-quality combined visualization for:
(a) Community-level mean opinion profiles across thematic domains
(b) Question-wise response variance / polarization across all 60 survey questions
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add current directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import THEME_COLORS

# Set publication style
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def generate_panel_a(df_profiles, ax=None, standalone=False):
    """
    Panel (a): Community-level mean opinion profiles across thematic domains.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    else:
        fig = ax.figure

    themes = ["Technology", "Education", "Society & Ethics", "Environment"]
    x = np.arange(len(themes))
    n_comm = len(df_profiles)
    width = 0.82 / n_comm

    comm_palette = sns.color_palette("tab10", n_comm)

    for idx, row in df_profiles.iterrows():
        cid = int(row["community_id"])
        scores = [
            row["technology_mean"],
            row["education_mean"],
            row["society_mean"],
            row["environment_mean"],
        ]
        offset = (idx - n_comm / 2) * width + width / 2
        label = f"Comm {cid} (N={int(row['size'])})"
        ax.bar(
            x + offset,
            scores,
            width=width,
            label=label,
            color=comm_palette[idx],
            edgecolor="#222222",
            linewidth=0.5,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(themes, fontsize=10.5, fontweight="bold", color="#111111")
    ax.set_ylabel("Mean Likert Score (Scale: -2 to +2)", fontsize=11, fontweight="bold", color="#111111")
    ax.set_ylim(-0.6, 2.05)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.8)

    # Subtle grid
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, color="#888888")
    ax.set_axisbelow(True)

    ax.set_title("Community-Level Mean Opinion Profiles", fontsize=12, fontweight="bold", pad=10)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8.2,
        ncol=2,
    )

    if standalone:
        fig.tight_layout()
        return fig
    return ax


def generate_panel_b(df_q_metrics, ax=None, standalone=False):
    """
    Panel (b): Question-wise response variance across all 60 questions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    else:
        fig = ax.figure

    # Sort all 60 questions by variance_polarization ascending
    df_sorted = df_q_metrics.sort_values(by="variance_polarization", ascending=True).reset_index(drop=True)

    x = np.arange(len(df_sorted))
    variances = df_sorted["variance_polarization"].values
    codes = df_sorted["code"].values
    categories = df_sorted["category"].values

    bar_colors = [THEME_COLORS.get(cat, "#7f8c8d") for cat in categories]

    # Plot bars
    bars = ax.bar(
        x,
        variances,
        width=0.75,
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.4,
        alpha=0.88,
    )

    # Average variance reference line
    mean_var = np.mean(variances)
    ax.axhline(
        mean_var,
        color="#c0392b",
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        label=rf"Mean Variance ($\mu = {mean_var:.2f}$)",
    )

    # Carefully staggered annotations to avoid any overlap
    # Consensus: E15 (idx 0), V13 (idx 1), S05 (idx 2)
    ax.annotate(
        "E15\n(0.25)",
        xy=(0, variances[0]),
        xytext=(-0.8, 0.52),
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.9),
    )
    ax.annotate(
        "V13\n(0.31)",
        xy=(1, variances[1]),
        xytext=(1.8, 0.68),
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.9),
    )
    ax.annotate(
        "S05\n(0.35)",
        xy=(2, variances[2]),
        xytext=(4.2, 0.84),
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.9),
    )

    # Divisive: E02 (idx 57), E03 (idx 58), E04 (idx 59)
    ax.annotate(
        "E02\n(1.25)",
        xy=(57, variances[57]),
        xytext=(55.2, 1.44),
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9),
    )
    ax.annotate(
        "E03\n(1.26)",
        xy=(58, variances[58]),
        xytext=(57.6, 1.56),
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9),
    )
    ax.annotate(
        "E04\n(1.36)",
        xy=(59, variances[59]),
        xytext=(60.0, 1.44),
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=90, fontsize=6.8, fontweight="medium")
    ax.set_xlabel(r"Survey Question Code (Ordered by Response Variance $\sigma^2$)", fontsize=11, fontweight="bold", color="#111111", labelpad=8)
    ax.set_ylabel(r"Response Variance ($\sigma^2$)", fontsize=11, fontweight="bold", color="#111111")
    ax.set_ylim(0, 1.75)

    ax.yaxis.grid(True, linestyle=":", alpha=0.5, color="#888888")
    ax.set_axisbelow(True)

    ax.set_title("Question-Level Response Variance and Polarization", fontsize=12, fontweight="bold", pad=10)

    # Custom legend for domains + mean line
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=THEME_COLORS["Technology"], edgecolor="#222222", label="Technology", linewidth=0.5),
        Patch(facecolor=THEME_COLORS["Education"], edgecolor="#222222", label="Education", linewidth=0.5),
        Patch(facecolor=THEME_COLORS["Society & Ethics"], edgecolor="#222222", label="Society & Ethics", linewidth=0.5),
        Patch(facecolor=THEME_COLORS["Environment"], edgecolor="#222222", label="Environment", linewidth=0.5),
        plt.Line2D([0], [0], color="#c0392b", linestyle="--", linewidth=1.0, label=rf"Mean $\sigma^2 = {mean_var:.2f}$"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8.0,
        ncol=2,
    )

    if standalone:
        fig.tight_layout()
        return fig
    return ax


def create_combined_figure(df_profiles, df_q_metrics, output_path):
    """
    Creates a single unified two-panel figure containing:
    (a) Community Opinion Profiles
    (b) Question-wise Response Variance
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300, gridspec_kw={"width_ratios": [1, 1.35]})

    generate_panel_a(df_profiles, ax=ax1)
    generate_panel_b(df_q_metrics, ax=ax2)

    # Add Panel Labels (a) and (b)
    ax1.text(-0.10, 1.05, "(a)", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="top")
    ax2.text(-0.08, 1.05, "(b)", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="top")

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined figure to: {output_path}")


if __name__ == "__main__":
    profiles_path = "output/tables/community_profiles.csv"
    q_metrics_path = "output/tables/question_centralities.csv"

    df_profiles = pd.read_csv(profiles_path)
    df_q_metrics = pd.read_csv(q_metrics_path)

    # Output paths
    figures_dir = "output/figures"
    report_figures_dir = "report/figures"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(report_figures_dir, exist_ok=True)

    comb_out = os.path.join(figures_dir, "fig5_community_and_question_polarization.png")
    comb_report_out = os.path.join(report_figures_dir, "fig5_community_and_question_polarization.png")

    create_combined_figure(df_profiles, df_q_metrics, comb_out)
    create_combined_figure(df_profiles, df_q_metrics, comb_report_out)

    # Also save standalone panels for modularity
    fig_a = generate_panel_a(df_profiles, standalone=True)
    fig_a.savefig(os.path.join(figures_dir, "fig5a_community_opinion_profiles.png"), dpi=300, bbox_inches="tight")
    fig_a.savefig(os.path.join(report_figures_dir, "fig5a_community_opinion_profiles.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_a)

    fig_b = generate_panel_b(df_q_metrics, standalone=True)
    fig_b.savefig(os.path.join(figures_dir, "fig5b_question_variance.png"), dpi=300, bbox_inches="tight")
    fig_b.savefig(os.path.join(report_figures_dir, "fig5b_question_variance.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_b)

    print("All panel and combined figures generated successfully.")
