"""Data utilities for optimizer: dataset conversion, splitting, saving."""

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import dspy
import yaml


def extract_value(annotation: dict, json_path: str):
    """Extract value from annotation using json_path format: TYPE::section::field."""
    try:
        parts = json_path.split('::')
        if len(parts) < 3 or parts[1] not in annotation:
            return None
        
        path_type, section, field = parts[0], parts[1], parts[2]
        section_data = annotation[section]
        
        if path_type == "STATIC":
            return section_data.get('static_fields', {}).get(field)
        elif path_type == "TABLE":
            return section_data.get('tables', {}).get(field, [])
    except:
        return None


def mapper_to_dspy(
    mapper: dict | Path,
    config_path: Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Convert mapper dict to DSPy datasets.
    
    Args:
        mapper: Dict with {ann_path: text} or path to pickled dict
        config_path: Path to YAML config with field definitions
        train_frac: Training set fraction
        val_frac: Validation set fraction
        
    Returns:
        (trainset, valset, testset)
    """
    # Load mapper if path provided
    if isinstance(mapper, Path):
        with open(mapper, 'rb') as f:
            mapper = pickle.load(f)
    
    # Load field config
    with open(config_path) as f:
        fields = yaml.safe_load(f).get('fields', {})
    
    # Build examples
    examples = []
    skipped = 0
    
    for ann_path, content in mapper.items():
        ann_path = Path(ann_path)
        
        if not ann_path.exists() or not content:
            skipped += 1
            continue
        
        # Load annotation JSON
        with open(ann_path) as f:
            annotation = json.load(f)
        
        # Build example dict
        d = {"document_text": str(content)}
        
        for name, config in fields.items():
            if path := config.get('json_path'):
                value = extract_value(annotation, path)
                d[name] = json.dumps(value) if isinstance(value, list) else value
        
        examples.append(dspy.Example(**d).with_inputs("document_text"))
    
    # Split datasets
    n = len(examples)
    t_end = int(n * train_frac)
    v_end = t_end + int(n * val_frac)
    
    trainset = examples[:t_end]
    valset = examples[t_end:v_end]
    testset = examples[v_end:]
    
    print(f"Dataset: {len(examples)} examples ({skipped} skipped)")
    print(f"  Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    
    return trainset, valset, testset


def json_to_dspy(
    json_path: Path,
    text_col: str = "document_text",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Convert JSON dataset to DSPy datasets.
    
    If dataset has _split markers (from gen_data.py), uses those splits.
    Otherwise, creates new splits based on train_frac/val_frac.
    
    Args:
        json_path: Path to JSON file (list of dicts)
        text_col: Column containing document text
        train_frac: Training set fraction (used only if no _split markers)
        val_frac: Validation set fraction (used only if no _split markers)
        
    Returns:
        (trainset, valset, testset)
    """
    with open(json_path) as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON dataset must be a list of records")
    
    # Check if dataset has _split markers
    has_splits = any('_split' in record for record in data)
    
    examples = []
    
    for record in data:
        if not isinstance(record, dict):
            continue
            
        d = {"document_text": str(record.get(text_col, ""))}
        
        # Copy all other fields (including _split if present)
        for key, val in record.items():
            if key == text_col:
                continue
            
            # Parse JSON strings if needed
            if isinstance(val, str) and val.startswith('['):
                try:
                    d[key] = json.loads(val)
                except json.JSONDecodeError:
                    d[key] = val
            else:
                d[key] = val
        
        examples.append(dspy.Example(**d).with_inputs("document_text"))
    
    if has_splits:
        # Use existing splits from gen_data.py
        trainset = [ex for ex in examples if getattr(ex, '_split', None) == 'train']
        valset = [ex for ex in examples if getattr(ex, '_split', None) == 'val']
        testset = [ex for ex in examples if getattr(ex, '_split', None) == 'test']
        
        print(f"\nDataset: {len(examples)} examples (using existing splits)")
        print(f"  Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    else:
        # Create new splits
        n = len(examples)
        t_end = int(n * train_frac)
        v_end = t_end + int(n * val_frac)
        
        trainset = examples[:t_end]
        valset = examples[t_end:v_end]
        testset = examples[v_end:]
        
        print(f"\nDataset: {n} examples (creating new splits)")
        print(f"  Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    
    return trainset, valset, testset


def save_datasets(trainset: List[dspy.Example], valset: List[dspy.Example], 
                  testset: List[dspy.Example], output_dir: Path):
    """Save train/val/test splits to pickle files."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / "trainset.pkl", "wb") as f:
        pickle.dump(trainset, f)
    with open(output_dir / "valset.pkl", "wb") as f:
        pickle.dump(valset, f)
    with open(output_dir / "testset.pkl", "wb") as f:
        pickle.dump(testset, f)
    
    print(f"✓ Saved datasets to {output_dir}")


def load_datasets(output_dir: Path) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Load train/val/test splits from pickle files."""
    with open(output_dir / "trainset.pkl", "rb") as f:
        trainset = pickle.load(f)
    with open(output_dir / "valset.pkl", "rb") as f:
        valset = pickle.load(f)
    with open(output_dir / "testset.pkl", "rb") as f:
        testset = pickle.load(f)
    
    return trainset, valset, testset