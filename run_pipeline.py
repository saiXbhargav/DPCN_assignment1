"""
run_pipeline.py
Master execution script for DPCN Assignment 1: Opinion Network Formation.
Executes data preprocessing, Model 1 (Student Network), and Model 2 (Question Network),
saving all figures, metrics tables, and execution logs.
"""

import os
import sys
import argparse

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure src is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.preprocessing import load_and_clean_survey_data
from src.model1_student_network import run_model1_pipeline
from src.model2_question_network import run_model2_pipeline


def main():
    parser = argparse.ArgumentParser(description="DPCN Assignment 1 Network Analysis Pipeline")
    parser.add_argument("--csv", type=str, default=os.path.join(SCRIPT_DIR, "Survey_Results_UC.csv"),
                        help="Path to survey CSV file")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "output"),
                        help="Output directory for figures and tables")
    parser.add_argument("--tau_student", type=float, default=0.35,
                        help="Similarity threshold for student network (default: 0.35)")
    parser.add_argument("--tau_question", type=float, default=0.25,
                        help="Correlation threshold for question network (default: 0.25)")
    args = parser.parse_args()

    print("=================================================================")
    print("      DPCN ASSIGNMENT 1: OPINION NETWORK FORMATION PIPELINE      ")
    print("=================================================================")

    # Step 1: Preprocessing
    print("\n[Step 1/3] Loading and Preprocessing Survey Data...")
    df_numeric, df_metadata, cleaning_stats = load_and_clean_survey_data(
        args.csv, scale="zero_centered", min_answers=30, impute_strategy="median"
    )

    print(f" -> Initial respondents: {cleaning_stats['total_initial_respondents']}")
    print(f" -> Empty rows removed: {cleaning_stats['completely_empty_rows_dropped']} {cleaning_stats['empty_respondent_ids']}")
    print(f" -> Severely incomplete removed (<30 answers): {cleaning_stats['severely_incomplete_dropped']} {cleaning_stats['incomplete_respondent_ids']}")
    print(f" -> Final clean respondents: {cleaning_stats['final_active_respondents']}")
    print(f" -> Questions analyzed: {cleaning_stats['total_questions']} across 4 themes")

    # Step 2: Model 1 - Student Opinion Network
    print("\n[Step 2/3] Executing Model 1: Student-Student Opinion Network...")
    G_s, df_nodes, global_metrics, df_profiles = run_model1_pipeline(
        df_numeric, df_metadata, args.output, threshold=args.tau_student
    )

    # Step 3: Model 2 - Question Concept Network
    print("\n[Step 3/3] Executing Model 2: Question-Question Concept Network...")
    G_q, df_corr, df_q_metrics = run_model2_pipeline(
        df_numeric, df_metadata, args.output, threshold=args.tau_question
    )

    print("\n=================================================================")
    print("                       PIPELINE SUMMARY                          ")
    print("=================================================================")
    print(f"Student Network (tau={args.tau_student}):")
    print(f"  - Nodes: {global_metrics['num_nodes']}, Edges: {global_metrics['num_edges']}")
    print(f"  - Average Degree <k>: {global_metrics['average_degree']}")
    print(f"  - Graph Density: {global_metrics['density']}")
    print(f"  - Empirical Clustering (C): {global_metrics['avg_clustering_coeff']}")
    print(f"  - ER Random Graph Clustering (C_ER): {global_metrics['er_clustering_simulated_mean']}")
    print(f"  - GCC Avg. Path Length (L): {global_metrics['avg_path_length_gcc']}")
    print(f"  - ER Random Graph Path Length (L_ER): {global_metrics['er_path_length_simulated_mean']}")
    print(f"  - Small-World Index (Sigma): {global_metrics['small_world_index_sigma']:.3f} (Values > 1 indicate small-world structure)")
    print(f"  - Detected Communities: {len(df_profiles)} distinct cohorts")

    top_bridge_student = df_nodes.sort_values(by="betweenness_centrality", ascending=False).index[0]
    top_degree_student = df_nodes.sort_values(by="degree", ascending=False).index[0]
    print(f"  - Top Consensus Student (Highest Degree): ID {top_degree_student} (Degree={df_nodes.loc[top_degree_student, 'degree']})")
    print(f"  - Top Bridge Student (Highest Betweenness): ID {top_bridge_student} (Betweenness={df_nodes.loc[top_bridge_student, 'betweenness_centrality']:.4f})")

    print(f"\nQuestion Network (tau={args.tau_question}):")
    print(f"  - Nodes: {G_q.number_of_nodes()}, Edges: {G_q.number_of_edges()}")
    top_bridge_q = df_q_metrics.iloc[0]
    print(f"  - Top Thematic Bridge Question: {top_bridge_q['code']} ({top_bridge_q['category']}) - Betweenness: {top_bridge_q['betweenness_centrality']:.4f}")
    print(f"    Text: \"{top_bridge_q['question_text']}\"")

    most_divisive = df_q_metrics.sort_values(by="variance_polarization", ascending=False).iloc[0]
    highest_consensus = df_q_metrics.sort_values(by="variance_polarization", ascending=True).iloc[0]
    print(f"  - Most Divisive / Controversial Question: {most_divisive['code']} (Variance={most_divisive['variance_polarization']})")
    print(f"    Text: \"{most_divisive['question_text']}\"")
    print(f"  - Highest Consensus Question: {highest_consensus['code']} (Variance={highest_consensus['variance_polarization']}, Mean={highest_consensus['mean_score']})")
    print(f"    Text: \"{highest_consensus['question_text']}\"")

    print("\nAll figures exported to: output/figures/")
    print("All tables exported to:  output/tables/")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
