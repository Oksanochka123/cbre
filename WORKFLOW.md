# Field Extraction Pipeline - Complete Workflow

This document provides a step-by-step guide for the complete field extraction pipeline, from data preparation to evaluation.

## Overview

The field extraction pipeline consists of 5 main stages:

0. **Extract Ground Truth** - Extract structured data from Excel lease abstracts (ground truth)
1. **Optimize Prompts** - Use DSPy/GEPA to optimize field extraction prompts
2. **Export Prompts** - Extract best-performing prompts from optimization logs
3. **Run Inference** - Use optimized prompts to extract fields from lease documents
4. **Evaluate Results** - Compare predictions against ground truth and generate metrics

---

## Prerequisites

### 1. Environment Setup (SEE README)
### 2. Required Files

- `configs/lease_mapping_config.json` - Excel cell mapping configuration for ground truth extraction
- `configs/fields_config.yaml` - Field configuration with types, matchers, json_ref paths
- `.env` - Contains `OPENAI_API_KEY`
- Raw lease documents: PDFs and Excel abstracts (`.xlsm` files)
- Lease documents in VLM text format (`*_vlm.txt`)

---

## Stage 0: Extract Ground Truth (Data Preparation)

**Purpose**: Extract structured ground truth data from Excel lease abstracts into JSON format.

### Input Structure

Your raw data should be organized with lease documents in subfolders:

```
data/raw/
├── Lease_001_Name/
│   ├── lease_document.pdf
│   ├── lease_amendment.pdf
│   └── lease_abstract.xlsm          # ← Excel ground truth
├── Lease_002_Name/
│   ├── lease_document.pdf
│   └── lease_abstract.xlsm
└── ... (other lease folders)
```

**Note**: Each subfolder contains:
- PDF lease documents (for reference/future VLM processing)
- One Excel abstract file (`.xlsm`) with structured lease data

### Configuration File

The [configs/lease_mapping_config.json](configs/lease_mapping_config.json) defines how to extract data from Excel sheets:

```json
{
  "Gen Info 1": {
    "static_fields": {
      "Gen Info 1|Property Information|Property Name": "B10",
      "Gen Info 1|Lease Information|Lease Commence": "B18",
      "Gen Info 1|Lease Information|Expiration Date": "H22"
    },
    "tables": {
      "Gen Info 1|Premise Information": {
        "start_row": 32,
        "columns": {
          "floor_number": "B",
          "unit_number": "C",
          "rentable_leaseable_sf": "D"
        }
      }
    }
  },
  "Financial Info": {
    "static_fields": {
      "Financial Info|Rent Information|Monthly Rent": "B10"
    }
  }
}
```

**Configuration Structure**:
- **Sheet names** (e.g., "Gen Info 1", "Financial Info") - must match Excel sheet names exactly
- **static_fields** - Single-cell values with cell references (e.g., "B10")
- **tables** - Multi-row data with start row and column mappings
- **Conditional cells** - Use `if` conditions to handle dynamic cell locations

### Command

```bash
python scripts/extract_lease_ground_truth.py \
    data/raw \
    configs/lease_mapping_config.json \
    data/extracted_fields
```

### Arguments

| Position | Description | Example |
|----------|-------------|---------|
| 1 | Input folder with subfolders containing `.xlsm` files | `data/raw` |
| 2 | Path to lease mapping config JSON | `configs/lease_mapping_config.json` |
| 3 | Output folder for extracted JSON files | `data/extracted_fields` |

### Output Structure

```
data/extracted_fields/
├── Lease_001_Name/
│   └── lease_abstract.json         # Extracted ground truth
├── Lease_002_Name/
│   └── lease_abstract.json
└── ... (mirrors input structure)
```

### Output JSON Format

Each extracted JSON file contains structured data organized by sheet:

```json
{
  "Gen Info 1": {
    "static_fields": {
      "Gen Info 1|Property Information|Property Name": "Activity Business Center",
      "Gen Info 1|Lease Information|Lease Commence": "2023-12-15",
      "Gen Info 1|Lease Information|Expiration Date": "2028-12-14"
    },
    "tables": {
      "Gen Info 1|Premise Information": [
        {
          "floor_number": "1",
          "unit_number": "101",
          "rentable_leaseable_sf": 1500
        },
        {
          "floor_number": "1",
          "unit_number": "102",
          "rentable_leaseable_sf": 1200
        }
      ]
    }
  },
  "Financial Info": {
    "static_fields": {
      "Financial Info|Rent Information|Monthly Rent": 5000.00
    },
    "tables": {}
  }
}
```



After extraction, you'll typically:
1. **Prepare VLM text files** - Convert PDFs to text format (`*_vlm.txt`)
2. **Move to interim** - Combine ground truth JSON with VLM text files:



- Add VLM text files (from separate PDF processing)
- Place *_vlm.txt files in corresponding data/interim/{lease_name}/ folders


3. **Build training dataset** - Create CSV with annotations for optimization


---

## Stage 1: Optimize Prompts

**Purpose**: Use DSPy/GEPA optimization to find the best prompts for each field.

### Command

```bash
python scripts/optimize_fields.py \
    --config configs/fields_config.yaml \
    --data data/dataset_annotated.csv \
    --api_key $OPENAI_API_KEY \
    --output optimizer_logs/optimized_all \
    --train_frac 0.8 \
    --val_frac 0.1
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to fields configuration YAML | Required |
| `--data` | Path to training dataset CSV | Required |
| `--api_key` | OpenAI API key | Required |
| `--output` | Output directory for optimization logs | `./data/processed` |
| `--fields` | Specific fields to optimize (space-separated) | All fields |
| `--group` | Field group name from config | None |
| `--train_frac` | Fraction of data for training | `0.8` |
| `--val_frac` | Fraction of data for validation | `0.1` |
| `--no_logging` | Disable detailed logging | False |

### Output Structure

```
optimizer_logs/optimized_all/
├── logs/
│   ├── property_name/
│   │   ├── gepa_results_property_name.json
│   │   └── optimization_logs.txt
│   ├── lease_commence/
│   │   └── gepa_results_lease_commence.json
│   └── ... (other fields)
└── optimization_summary.json
```

### Example: Optimize Specific Fields

```bash
# Optimize only 3 specific fields
python scripts/optimize_fields.py \
    --config configs/fields_config.yaml \
    --data data/dataset_annotated.csv \
    --api_key $OPENAI_API_KEY \
    --output optimizer_logs/test_run \
    --fields property_name lease_commence rent_amount
```

### Example: Optimize Field Group

```bash
# Optimize a predefined group (if defined in config)
python scripts/optimize_fields.py \
    --config configs/fields_config.yaml \
    --data data/dataset_annotated.csv \
    --api_key $OPENAI_API_KEY \
    --output optimizer_logs/priority_fields \
    --group priority_fields
```

---

## Stage 2: Export Optimized Prompts

**Purpose**: Extract the best-performing prompt for each field from GEPA results.

### Command

```bash
python scripts/export_optimized_prompts.py \
    --input optimizer_logs/optimized_all \
    --output prompts/final_export \
    --config configs/fields_config.yaml
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to optimizer logs directory | Required |
| `--output` | Output directory for exported prompts | `prompts/{timestamp}` |
| `--config` | Path to fields config YAML | `configs/fields_config.yaml` |
| `--verbose`, `-v` | Enable verbose logging | False |

### Output Structure

```
prompts/final_export/
├── property_name.json
├── lease_commence.json
├── tenant_entity_type.json
└── ... (one file per optimized field)
```

### Output JSON Format

Each file contains:

```json
{
  "field_name": "property_name",
  "type": "string",
  "matcher": "StringMatcher",
  "json_ref": "STATIC::Gen Info 1::Gen Info 1|Property Information|Property Name",
  "params": {
    "threshold": 0.85
  },
  "instructions": "Task overview\n- Input: A single field named...",
  "instructions_length": 6548,
  "score": 0.510902057815743
}
```

### What It Does

1. Reads field configuration from YAML
2. For each field in config:
   - Looks for `gepa_results_{field_name}.json` in logs directory
   - Extracts the candidate with the best score
   - Combines field metadata with best prompt
   - Saves to individual JSON file
3. Prints summary of successful/failed/skipped fields

### Summary Output

```
================================================================================
SUMMARY
================================================================================
Total fields:    94
✓ Successful:    35
⊘ Skipped:       57  (no optimization results yet)
✗ Failed:        2

Failed fields: field1, field2
================================================================================
```

### Example: Auto-Timestamped Export

```bash
# Output will be in prompts/YYYYMMDD_HHMMSS/
python scripts/export_optimized_prompts.py \
    --input optimizer_logs/optimized_all
```

---

## Stage 3: Run Inference

**Purpose**: Use optimized prompts to extract fields from lease documents.

### Command

```bash
python scripts/run_inference.py \
    --prompts prompts/final_export \
    --data data/interim \
    --output data/predictions \
    --config configs/inference_config.yaml
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to inference config YAML | `configs/inference_config.yaml` |
| `--prompts` | Directory with exported prompts | From config |
| `--data` | Directory with lease documents | From config |
| `--output` | Output directory for predictions | From config |
| `--system-prompt` | Optional system prompt file | From config |

### Configuration File

[configs/inference_config.yaml](configs/inference_config.yaml):

```yaml
llm:
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 4096
  timeout: 60

retry:
  max_attempts: 3
  initial_delay: 1.0
  backoff_factor: 2.0

processing:
  skip_existing: true  # Skip leases with existing output
  save_intermediate: true

paths:
  prompts_dir: "prompts/final_export"
  data_dir: "data/interim"
  output_dir: "data/predictions"
```

### Input Structure

```
data/interim/
├── Lease_001/
│   ├── document1_vlm.txt
│   ├── document2_vlm.txt
│   ├── document_meta.json  # (excluded)
│   └── ground_truth.json   # (optional)
└── Lease_002/
    └── ...
```

### Output Structure

```
data/predictions/
├── Lease_001/
│   └── predicted_fields.json
├── Lease_002/
│   └── predicted_fields.json
└── fields_config.yaml  # Copy of config for reference
```

### What It Does

1. Loads all optimized prompts from prompts directory
2. Lists all lease folders in data directory
3. For each lease:
   - Loads all `*_vlm.txt` files and concatenates
   - For each field:
     - Builds inference prompt with instructions + document text
     - Calls LLM API (with retry logic)
     - Extracts response
   - Builds structured JSON output using `json_ref` paths
   - Saves to `predicted_fields.json`
4. Prints summary of successful/failed leases

### Summary Output

```
================================================================================
INFERENCE COMPLETE
================================================================================
Total leases:     7
✓ Successful:     6
✗ Failed:         1
Output directory: /path/to/data/predictions
================================================================================
```

### Example: Test with Single Lease

```bash
# Create test folder with one lease
mkdir -p data/test_lease
cp -r "data/interim/Lease_001" data/test_lease/

# Run inference
python scripts/run_inference.py \
    --data data/test_lease \
    --output data/test_predictions
```

### Example: Process Subset

```bash
# Create subset
mkdir -p data/interim_subset
cp -r data/interim/Lease_00{1,2,3} data/interim_subset/

# Run inference
python scripts/run_inference.py \
    --data data/interim_subset \
    --output data/predictions_subset
```
---

## Stage 4: Evaluate Predictions

**Purpose**: Compare predictions against ground truth and generate accuracy metrics.

### Command

```bash
python scripts/evaluate_predictions.py \
    --ground-truth data/interim \
    --predictions data/predictions \
    --output results/evaluation.csv \
    --summary results/evaluation_summary.txt
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ground-truth` | Directory with ground truth JSONs | `data/interim` |
| `--predictions` | Directory with prediction JSONs | Required |
| `--config` | Path to fields config YAML | `configs/fields_config.yaml` |
| `--output` | Path to output CSV file | `results/evaluation.csv` |
| `--summary` | Path to summary text file | Same as CSV with `.txt` |
| `--log-file` | Optional log file path | None (console only) |

### What It Does

1. Loads field configuration (matchers, json_refs, params)
2. Creates field-specific matchers for each field
3. For each lease:
   - Loads ground truth JSON from interim folder
   - Loads prediction JSON from predictions folder
   - For each field:
     - Extracts value using `json_ref` path
     - Compares using appropriate matcher
     - Calculates score (0.0 to 1.0)
4. Aggregates per-field statistics
5. Generates CSV report and summary

### Output: CSV Report

[results/evaluation.csv](results/):

```csv
field_name,accuracy,correct_docs,total_docs,json_ref,matcher
property_name,0.8500,17,20,STATIC::Gen Info 1::...,StringMatcher
lease_commence,0.9500,19,20,STATIC::Gen Info 1::...,DateMatcher
tenant_entity_type,1.0000,20,20,STATIC::Gen Info 1::...,EnumMatcher
...
```

### Output: Summary Report

[results/evaluation_summary.txt](results/):

```
================================================================================
FIELD EXTRACTION EVALUATION SUMMARY
================================================================================
Generated: 2025-11-17 03:07:01

OVERALL STATISTICS
--------------------------------------------------------------------------------
Total leases processed:     100
Leases skipped:             5
Total fields evaluated:     94
Total field evaluations:    9400
Total correct:              7520
Overall accuracy:           0.8000 (80.00%)

TOP 10 BEST PERFORMING FIELDS
--------------------------------------------------------------------------------
Field                                      Accuracy   Correct/Total
--------------------------------------------------------------------------------
billing_state                                1.0000          100/100
tenant_entity_type                           0.9900           99/100
...

BOTTOM 10 WORST PERFORMING FIELDS
--------------------------------------------------------------------------------
Field                                      Accuracy   Correct/Total
--------------------------------------------------------------------------------
complex_field_1                              0.4500           45/100
complex_field_2                              0.5200           52/100
...
```

### Matching Logic

- **Correct**: Matcher score ≥ 0.95 (accounts for fuzzy matching)
- **Both None**: Score = 1.0 (consistent)
- **Hallucination** (GT None, Pred exists): Score = 0.0
- **Missing** (GT exists, Pred None): Score = 0.0

### Example: Evaluate Test Set

```bash
python scripts/evaluate_predictions.py \
    --ground-truth data/test_inference \
    --predictions data/test_predictions \
    --output results/test_eval.csv
```

### Example: With Logging

```bash
python scripts/evaluate_predictions.py \
    --ground-truth data/interim \
    --predictions data/predictions \
    --output results/evaluation.csv \
    --summary results/summary.txt \
    --log-file logs/evaluation_$(date +%Y%m%d_%H%M%S).log
```


---

## Complete Workflow Example

Here's the complete end-to-end workflow:

```bash
#!/bin/bash
# Complete field extraction pipeline

# Activate environment
source venv/bin/activate

# Set timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting field extraction pipeline at $TIMESTAMP"

# Stage 0: Extract Ground Truth (run once for new data)
echo "Stage 0: Extracting ground truth from Excel abstracts..."
python scripts/extract_lease_ground_truth.py \
    data/raw \
    configs/lease_mapping_config.json \
    data/extracted_fields

# Move to interim folder and prepare for optimization
echo "Preparing interim data..."
for lease_dir in data/extracted_fields/*/; do
    lease_name=$(basename "$lease_dir")
    mkdir -p "data/interim/$lease_name"
    cp "$lease_dir"/*.json "data/interim/$lease_name/"
done

# Stage 1: Optimize Prompts (run once or when training data changes)
echo "Stage 1: Optimizing prompts..."
python scripts/optimize_fields.py \
    --config configs/fields_config.yaml \
    --data data/dataset_annotated.csv \
    --api_key $OPENAI_API_KEY \
    --output optimizer_logs/run_$TIMESTAMP \
    --train_frac 0.8 \
    --val_frac 0.1

# Stage 2: Export Prompts
echo "Stage 2: Exporting optimized prompts..."
python scripts/export_optimized_prompts.py \
    --input optimizer_logs/run_$TIMESTAMP \
    --output prompts/export_$TIMESTAMP

# Stage 3: Run Inference
echo "Stage 3: Running inference on lease documents..."
python scripts/run_inference.py \
    --prompts prompts/export_$TIMESTAMP \
    --data data/interim \
    --output data/predictions_$TIMESTAMP

# Stage 4: Evaluate Results
echo "Stage 4: Evaluating predictions..."
python scripts/evaluate_predictions.py \
    --ground-truth data/interim \
    --predictions data/predictions_$TIMESTAMP \
    --output results/evaluation_$TIMESTAMP.csv \
    --summary results/summary_$TIMESTAMP.txt \
    --log-file logs/evaluation_$TIMESTAMP.log

echo "Pipeline complete! Results in results/evaluation_$TIMESTAMP.csv"
echo "Summary: results/summary_$TIMESTAMP.txt"
```

---

## Quick Reference

### First-Time Setup (With New Raw Data)

```bash
# 0. Extract ground truth from Excel abstracts
python scripts/extract_lease_ground_truth.py \
    data/raw \
    configs/lease_mapping_config.json \
    data/extracted_fields

# Prepare interim folder (combine with VLM text files)
# ... copy extracted JSON and VLM files to data/interim/ ...

# 1. Optimize prompts (one-time or when retraining)
python scripts/optimize_fields.py \
    --config configs/fields_config.yaml \
    --data data/dataset_annotated.csv \
    --api_key $OPENAI_API_KEY \
    --output optimizer_logs/optimized_all
```

### Typical Workflow (After Initial Optimization)

```bash
# 1. Export prompts from latest optimization
python scripts/export_optimized_prompts.py \
    --input optimizer_logs/optimized_all \
    --output prompts/latest

# 2. Run inference
python scripts/run_inference.py \
    --prompts prompts/latest \
    --output data/predictions

# 3. Evaluate
python scripts/evaluate_predictions.py \
    --predictions data/predictions \
    --output results/evaluation.csv
```

---
