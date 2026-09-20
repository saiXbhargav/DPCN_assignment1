# DPCN Assignment 1: LaTeX Report Project

This directory contains the complete, self-contained LaTeX source code and assets for the **Dynamics and Processes on Complex Networks (DPCN) Assignment 1** report.

## Directory Structure

```text
report/
├── main.tex               # Master LaTeX report document (~8 pages)
├── references.bib         # Academic bibliography in BibTeX format
├── figures/               # 300-DPI high-resolution analytical figures
│   ├── fig1_percolation_analysis.png
│   ├── fig2_student_network_communities.png
│   ├── fig3_degree_distribution.png
│   ├── fig4_community_opinion_radar.png
│   ├── fig5_question_correlation_heatmap.png
│   ├── fig6_question_thematic_network.png
│   └── fig7_question_percolation_analysis.png
├── tables/                # Modular LaTeX tables using booktabs
│   ├── tab1_global_metrics.tex
│   ├── tab2_student_centralities.tex
│   ├── tab3_community_profiles.tex
│   ├── tab4_question_polarization.tex
│   └── tab5_contributions.tex
├── main.pdf               # Compiled high-resolution PDF report (8 pages)
└── README.md              # Build instructions and project documentation
```

## Compilation Instructions

The project can be built using standard TeX distributions (`TeX Live`, `MacTeX`, or `MiKTeX`) with `pdflatex` and `bibtex`:

```bash
cd report/
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or using `latexmk`:

```bash
cd report/
latexmk -pdf main.tex
```

## Required LaTeX Packages

- `geometry` (page margins and layout)
- `amsmath, amssymb, amsfonts` (mathematical formulas)
- `booktabs` (professional tables)
- `graphicx, caption, subcaption` (figures and multi-panel subfigures)
- `hyperref` (hyperlinks and PDF metadata)
- `microtype` (font expansion and kerning)
- `tikz` (pipeline execution workflow diagram)
- `cite` (citation management)
- `enumitem, titlesec` (compact spacing and headings)
