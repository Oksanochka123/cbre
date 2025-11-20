import json
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple


def analyze_optimization_run(folder_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze optimization run results from a folder containing gepa_results_*.json files.

    Args:
        folder_path: Path to the optimization run folder (e.g., data/interim/test_optimization_run)

    Returns:
        Tuple of (DataFrame with analysis, dict with num_scores statistics)
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    results = []
    num_scores_analysis = {
        'min': None,
        'max': None,
        'mean': None,
        'fields': {}
    }

    # Find all gepa_results_*.json files recursively
    gepa_files = list(folder_path.rglob("gepa_results_*.json"))

    if not gepa_files:
        raise FileNotFoundError(f"No gepa_results_*.json files found in {folder_path}")

    print(f"Found {len(gepa_files)} gepa_results files\n")

    # Process each gepa_results file
    for gepa_file in sorted(gepa_files):
        try:
            with open(gepa_file, 'r') as f:
                data = json.load(f)

            # Extract field information
            field_name = data.get('field_name', 'unknown')
            field_type = data.get('field_type', 'unknown')

            # Get best score and index
            val_aggregate_scores = data.get('val_aggregate_scores', [])
            best_idx = data.get('best_idx', -1)
            num_scores = len(val_aggregate_scores)

            # Get best score
            if 0 <= best_idx < len(val_aggregate_scores):
                best_score = val_aggregate_scores[best_idx]
            else:
                best_score = None

            results.append({
                'column': field_name,
                'data_type': field_type,
                'best_score': best_score,
                'best_idx': best_idx,
                'num_scores': num_scores,
                'all_scores': val_aggregate_scores,
                'file_path': str(gepa_file)
            })

            # Track num_scores statistics
            num_scores_analysis['fields'][field_name] = num_scores

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {gepa_file}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {gepa_file}: {e}")

    # Calculate num_scores statistics
    if results:
        num_scores_list = [r['num_scores'] for r in results]
        num_scores_analysis['min'] = min(num_scores_list)
        num_scores_analysis['max'] = max(num_scores_list)
        num_scores_analysis['mean'] = sum(num_scores_list) / len(num_scores_list)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Select main columns
    df = df[['column', 'data_type', 'best_score', 'num_scores', 'best_idx']]

    # Sort by best_score descending
    df = df.sort_values('best_score', ascending=False, na_position='last').reset_index(drop=True)

    return df, num_scores_analysis


def print_analysis(df: pd.DataFrame, num_scores_analysis: Dict):
    """Print formatted analysis results."""
    print("="*80)
    print("OPTIMIZATION RUN ANALYSIS")
    print("="*80)
    print()

    print("Summary DataFrame:")
    print(df.to_string(index=False))
    print()

    print("="*80)
    print("NUM_SCORES STATISTICS")
    print("="*80)
    print(f"Minimum candidates: {num_scores_analysis['min']}")
    print(f"Maximum candidates: {num_scores_analysis['max']}")
    print(f"Average candidates: {num_scores_analysis['mean']:.2f}")
    print(f"Total fields: {len(num_scores_analysis['fields'])}")
    print()

    print("Candidates per field:")
    for field, count in sorted(num_scores_analysis['fields'].items()):
        print(f"  {field}: {count}")
    print()

    # Score statistics
    print("="*80)
    print("BEST SCORE STATISTICS")
    print("="*80)
    print(f"Highest score: {df['best_score'].max():.4f}")
    print(f"Lowest score: {df['best_score'].min():.4f}")
    print(f"Mean score: {df['best_score'].mean():.4f}")
    print(f"Median score: {df['best_score'].median():.4f}")
    print(f"Std dev: {df['best_score'].std():.4f}")
    print()

    # Distribution
    print("Score distribution:")
    print(df['best_score'].describe().to_string())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_optimization.py <path_to_optimization_folder>")
        print("Example: python analyze_optimization.py data/interim/test_optimization_run")
        sys.exit(1)

    folder_path = sys.argv[1]

    try:
        df, num_scores_analysis = analyze_optimization_run(folder_path)
        print_analysis(df, num_scores_analysis)

        # Save to CSV
        output_path = Path(folder_path) / "optimization_analysis.csv"
        df.to_csv(output_path, index=False)
        print(f"\nDataFrame saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
