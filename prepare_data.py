import argparse
import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy.sparse
import joblib
from typing import Tuple, Optional

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for search relevance prediction.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw hybrid_output CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store processed outputs")
    parser.add_argument("--valid_size", type=float, default=0.1, help="Validation set ratio (default 0.1)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set ratio (default 0.1)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default 42)")
    return parser.parse_args()

def clean_text(text: str) -> str:
    """
    Strip HTML tags, normalize whitespace, and lowercase text.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    return text.lower()

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Load CSV, handle missing values, and perform basic text cleaning.
    """
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Check for required columns
    required_cols = ['question', 'title']
    # 'relevance_label' might be 'relevance' based on common variations
    label_col = None
    if 'relevance_label' in df.columns:
        label_col = 'relevance_label'
    elif 'relevance' in df.columns:
        label_col = 'relevance'
    else:
        print("Error: Could not find relevance label column (expected 'relevance_label' or 'relevance')")
        sys.exit(1)
    
    # Ensure contents column exists
    if 'contents' not in df.columns:
        print("Warning: 'contents' column not found. Treating as empty strings.")
        df['contents'] = ""

    # Drop rows with missing critical info
    initial_len = len(df)
    df = df.dropna(subset=['question', 'title', label_col])
    print(f"Dropped {initial_len - len(df)} rows with missing question, title, or label.")

    # Rename label column to standard 'relevance_label' if needed
    if label_col != 'relevance_label':
        df = df.rename(columns={label_col: 'relevance_label'})

    # Convert label to float
    try:
        df['relevance_label'] = pd.to_numeric(df['relevance_label'], errors='coerce')
        df = df.dropna(subset=['relevance_label'])
    except Exception as e:
        print(f"Error converting labels to numeric: {e}")
        sys.exit(1)

    # Fill NaN contents with empty string
    df['contents'] = df['contents'].fillna("")

    # Text cleaning
    print("Cleaning text fields...")
    for col in ['question', 'title', 'contents']:
        df[col] = df[col].apply(clean_text)

    return df

def build_tfidf_features(df: pd.DataFrame) -> Tuple[scipy.sparse.spmatrix, TfidfVectorizer]:
    """
    Create combined text field and vectorizer.
    """
    print("Constructing combined text field...")
    # combined_text = question + " " + title + " " + contents
    df['combined_text'] = (
        df['question'] + " " + 
        df['title'] + " " + 
        df['contents']
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_text = vectorizer.fit_transform(df['combined_text'])
    return X_text, vectorizer

def save_split(
    X: scipy.sparse.spmatrix, 
    y: np.ndarray, 
    scores: np.ndarray,
    questions: pd.Series, 
    titles: pd.Series, 
    split_name: str, 
    output_dir: str
):
    """
    Save all artifacts for a single split.
    """
    print(f"Saving {split_name} split to {output_dir}...")
    
    # Save Sparse Matrix
    scipy.sparse.save_npz(os.path.join(output_dir, f"X_{split_name}_text.npz"), X)
    
    # Save Labels
    np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y)
    
    # Save Relevance Scores (Baseline)
    np.save(os.path.join(output_dir, f"relevance_score_{split_name}.npy"), scores)
    
    # Save Questions and Titles
    questions.to_csv(os.path.join(output_dir, f"questions_{split_name}.csv"), index=False, header=True)
    titles.to_csv(os.path.join(output_dir, f"titles_{split_name}.csv"), index=False, header=True)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load and Clean
    df = load_and_clean_data(args.input_path)
    
    # Check if relevance_score exists, if not create dummy or fail? 
    # Prompt implies it exists: "The columns ... include at least: ... relevance_score"
    if 'relevance_score' not in df.columns:
        print("Warning: 'relevance_score' column missing. Filling with zeros.")
        df['relevance_score'] = 0.0
    else:
        df['relevance_score'] = pd.to_numeric(df['relevance_score'], errors='coerce').fillna(0.0)

    # 2. Vectorization
    X_text, vectorizer = build_tfidf_features(df)
    y = df['relevance_label'].values
    scores = df['relevance_score'].values
    questions = df['question']
    titles = df['title']

    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(args.output_dir, "vectorizer.joblib"))

    # 3. Split
    # Indices for splitting
    indices = np.arange(X_text.shape[0])
    
    # First split: Train+Valid vs Test
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X_text, y, indices, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=None # Stratify might fail if regression labels are continuous/many classes. Safe to assume random.
    )
    
    # Adjust valid size relative to remaining data
    # valid_size is ratio of TOTAL. 
    # We want valid_ratio of temp such that valid_count approx valid_size * total
    # valid_prop_of_temp = valid_size / (1 - test_size)
    if (1 - args.test_size) == 0:
         raise ValueError("Test size cannot be 1.0")
         
    valid_ratio = args.valid_size / (1 - args.test_size)
    
    X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=valid_ratio,
        random_state=args.random_state
    )

    print(f"Data split: Train={X_train.shape[0]}, Valid={X_valid.shape[0]}, Test={X_test.shape[0]}")

    # Helper to slice metadata arrays using indices
    def get_metadata(idx):
        return scores[idx], questions.iloc[idx], titles.iloc[idx]

    scores_train, q_train, t_train = get_metadata(idx_train)
    scores_valid, q_valid, t_valid = get_metadata(idx_valid)
    scores_test, q_test, t_test = get_metadata(idx_test)

    # 4. Save
    save_split(X_train, y_train, scores_train, q_train, t_train, "train", args.output_dir)
    save_split(X_valid, y_valid, scores_valid, q_valid, t_valid, "valid", args.output_dir)
    save_split(X_test, y_test, scores_test, q_test, t_test, "test", args.output_dir)

    print("Processing complete.")

if __name__ == "__main__":
    main()
