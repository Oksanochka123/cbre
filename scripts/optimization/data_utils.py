"""Data utilities for optimizer: dataset conversion, splitting, saving."""

import pickle
from pathlib import Path
from typing import List, Tuple

import dspy
import pandas as pd

# Fields to skip (metadata)
SKIP_FIELDS = {
    "timestamp",
    "path",
    "media_reference",
    "media_item_rid",
    "path_normalized",
    "mapped_xlsx",
    "text_item_full",
    "xlsx_source",
    "source_file",
    "pdf_stats",
    "completeness_metric",
    "used_vlm_content",
    "content",
}


def df_to_dspy(
    df: pd.DataFrame,
    text_col: str = "content",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Convert DataFrame to DSPy datasets with train/val/test split.
    
    Args:
        df: Input DataFrame
        text_col: Column containing document text
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        
    Returns:
        tuple: (trainset, valset, testset)
    """
    all_examples = []

    for idx, row in df.iterrows():
        d = {"document_text": str(row.get(text_col, ""))}

        for col in df.columns:
            if col != text_col and col not in SKIP_FIELDS:
                val = row[col]

                # Handle NaN/None
                if pd.isna(val):
                    d[col] = None
                # Handle Timestamps
                elif isinstance(val, pd.Timestamp):
                    d[col] = val.strftime("%Y-%m-%d")
                # Keep as-is
                else:
                    d[col] = val

        example = dspy.Example(**d).with_inputs("document_text")
        all_examples.append(example)

    # Standard split
    n = len(all_examples)
    t_end = int(n * train_frac)
    v_end = t_end + int(n * val_frac)

    trainset = all_examples[:t_end]
    valset = all_examples[t_end:v_end]
    testset = all_examples[v_end:]

    print(f"\nDataset split:")
    print(f"  Total records: {len(df)}")
    print(f"  Training: {len(trainset)} ({train_frac*100:.0f}%)")
    print(f"  Validation: {len(valset)} ({val_frac*100:.0f}%)")
    print(f"  Test: {len(testset)} ({(1-train_frac-val_frac)*100:.0f}%)")

    return trainset, valset, testset


def save_datasets(trainset: List[dspy.Example], valset: List[dspy.Example], testset: List[dspy.Example], output_dir: Path):
    """Save train/val/test splits to pickle files.
    
    Args:
        trainset: Training examples
        valset: Validation examples
        testset: Test examples
        output_dir: Directory to save files
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "trainset.pkl", "wb") as f:
        pickle.dump(trainset, f)

    with open(output_dir / "valset.pkl", "wb") as f:
        pickle.dump(valset, f)

    with open(output_dir / "testset.pkl", "wb") as f:
        pickle.dump(testset, f)

    print(f"✓ Saved train/val/test splits to {output_dir}")


def load_datasets(output_dir: Path) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Load train/val/test splits from pickle files.
    
    Args:
        output_dir: Directory containing the pickle files
        
    Returns:
        tuple: (trainset, valset, testset)
    """
    with open(output_dir / "trainset.pkl", "rb") as f:
        trainset = pickle.load(f)

    with open(output_dir / "valset.pkl", "rb") as f:
        valset = pickle.load(f)

    with open(output_dir / "testset.pkl", "rb") as f:
        testset = pickle.load(f)

    return trainset, valset, testset