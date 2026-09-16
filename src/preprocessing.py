"""
preprocessing.py
Data cleaning, parsing, Likert-scale encoding, and metadata extraction
for DPCN Assignment 1: Opinion Network Formation.
"""

import os
import re
import numpy as np
import pandas as pd

# Mapping for 5-point Likert Scale
LIKERT_MAP_ZERO_CENTERED = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
    "No Comments": 0,  # Neutral default for no comments
}

LIKERT_MAP_ORDINAL = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
    "No Comments": 3,
}

THEME_NAMES = {
    "T": "Technology",
    "E": "Education",
    "S": "Society & Ethics",
    "V": "Environment",
}

THEME_COLORS = {
    "Technology": "#2b5c8f",      # Sophisticated slate blue
    "Education": "#2a9d8f",       # Teal green
    "Society & Ethics": "#e76f51",# Warm terracotta
    "Environment": "#457b9d",     # Deep cyan / sea green
}


def parse_question_columns(columns):
    """
    Extracts code (e.g. 'T01'), category ('Technology'), and full question text
    from the raw CSV column headers.
    """
    metadata = []
    for col in columns:
        # Match pattern like 'T01. Question text...'
        match = re.match(r"^([TESV]\d{2})\.\s*(.*)$", col.strip())
        if match:
            code = match.group(1)
            category_key = code[0]
            text = match.group(2).strip()
            category = THEME_NAMES.get(category_key, "General")
            metadata.append({
                "column_raw": col,
                "code": code,
                "category_key": category_key,
                "category": category,
                "text": text,
            })
    return pd.DataFrame(metadata)


def load_and_clean_survey_data(
    csv_path,
    scale="zero_centered",
    min_answers=30,
    impute_strategy="median"
):
    """
    Loads and cleans the survey CSV:
    1. Removes completely empty respondent rows.
    2. Drops respondents with fewer than `min_answers` answered questions.
    3. Maps Likert text to numerical values.
    4. Imputes remaining missing values with question median/mean.

    Returns:
        df_numeric: Cleaned DataFrame (index=student ID, columns=question codes)
        df_metadata: Question metadata DataFrame
        cleaning_stats: Dictionary containing data cleaning summary statistics
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Survey CSV not found at {csv_path}")

    # Read CSV with proper encoding
    df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")

    # The first column is the Response ID
    id_col = df_raw.columns[0]
    question_cols = df_raw.columns[1:]

    total_initial_rows = len(df_raw)

    # 1. Detect completely empty rows
    is_completely_empty = df_raw[question_cols].isnull().all(axis=1)
    empty_ids = df_raw.loc[is_completely_empty, id_col].tolist()

    df_cleaned = df_raw[~is_completely_empty].copy()

    # 2. Check answered count per student
    answered_count = df_cleaned[question_cols].notnull().sum(axis=1)
    partial_ids = df_cleaned.loc[answered_count < min_answers, id_col].tolist()

    # Filter out respondents who answered fewer than `min_answers` questions
    df_filtered = df_cleaned[answered_count >= min_answers].copy()
    retained_ids = df_filtered[id_col].tolist()

    # 3. Extract question metadata
    df_metadata = parse_question_columns(question_cols)
    col_rename_map = dict(zip(df_metadata["column_raw"], df_metadata["code"]))

    # 4. Map Likert scale
    mapping = LIKERT_MAP_ZERO_CENTERED if scale == "zero_centered" else LIKERT_MAP_ORDINAL
    df_sub = df_filtered[df_metadata["column_raw"]].copy()
    df_sub = df_sub.rename(columns=col_rename_map)

    # Replace string values with numerical scores
    df_numeric = df_sub.apply(lambda col: col.map(mapping))
    df_numeric.index = df_filtered[id_col].astype(str).str.strip()

    # 5. Imputation for remaining NaNs
    missing_before_impute = df_numeric.isnull().sum().sum()
    if impute_strategy == "median":
        medians = df_numeric.median()
        df_numeric = df_numeric.fillna(medians)
    elif impute_strategy == "mean":
        means = df_numeric.mean()
        df_numeric = df_numeric.fillna(means)
    else:
        df_numeric = df_numeric.fillna(0)

    cleaning_stats = {
        "total_initial_respondents": total_initial_rows,
        "completely_empty_rows_dropped": len(empty_ids),
        "empty_respondent_ids": empty_ids,
        "severely_incomplete_dropped": len(partial_ids),
        "incomplete_respondent_ids": partial_ids,
        "final_active_respondents": len(df_numeric),
        "total_questions": len(df_metadata),
        "scale_used": scale,
        "imputed_cells": int(missing_before_impute),
        "themes_distribution": df_metadata["category"].value_counts().to_dict(),
    }

    return df_numeric, df_metadata, cleaning_stats


if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "..", "Survey_Results_UC.csv")
    df_num, meta, stats = load_and_clean_survey_data(csv_file)
    print("Cleaned data shape:", df_num.shape)
    print("Cleaning stats:", stats)
