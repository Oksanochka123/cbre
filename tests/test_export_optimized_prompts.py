import json

import pytest
import yaml

from scripts.export_optimized_prompts import (
    export_prompt,
    extract_best_prompt,
    load_gepa_results,
    process_optimization_logs,
)


class TestLoadGepaResults:
    """Test GEPA results loading."""

    def test_load_valid_gepa_results(self, tmp_path):
        """Test loading valid GEPA results."""
        field_name = "property_name"
        logs_dir = tmp_path
        field_dir = logs_dir / field_name
        field_dir.mkdir()

        gepa_data = {
            "best_idx": 2,
            "candidate_instructions": [{"index": 2, "score": 0.95, "instructions": "Best instruction"}],
        }

        gepa_file = field_dir / f"gepa_results_{field_name}.json"
        with open(gepa_file, "w") as f:
            json.dump(gepa_data, f)

        result = load_gepa_results(field_name, logs_dir)

        assert result["best_idx"] == 2
        assert len(result["candidate_instructions"]) == 1

    def test_load_nonexistent_gepa_results(self, tmp_path):
        """Test loading GEPA results when the file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="GEPA results not found"):
            load_gepa_results("nonexistent_field", tmp_path)


class TestExtractBestPrompt:
    """Test extracting best prompt from GEPA results."""

    def test_extract_best_prompt_success(self):
        """Test successful extraction of best prompt."""
        field_name = "property_name"
        gepa_results = {
            "best_idx": 1,
            "candidate_instructions": [
                {"index": 0, "score": 0.75, "instructions": "First", "instruction_length": 100},
                {"index": 1, "score": 0.95, "instructions": "Best", "instruction_length": 150},
            ],
        }

        field_config = {
            "type": "string",
            "matcher": "StringMatcher",
            "json_ref": "Property|Name",
            "params": {"threshold": 0.85},
        }

        result = extract_best_prompt(field_name, gepa_results, field_config)

        assert result["field_name"] == "property_name"
        assert result["instructions"] == "Best"
        assert result["score"] == 0.95
        assert result["instructions_length"] == 150

    def test_extract_best_prompt_no_best_idx(self):
        """Test extraction when best_idx is missing."""
        with pytest.raises(ValueError, match="No best_idx found"):
            extract_best_prompt("field", {"candidate_instructions": []}, {})

    def test_extract_best_prompt_candidate_not_found(self):
        """Test extraction when the best_idx candidate is not found."""
        gepa_results = {
            "best_idx": 5,
            "candidate_instructions": [{"index": 0, "score": 0.75}],
        }

        with pytest.raises(ValueError, match="Could not find candidate"):
            extract_best_prompt("field", gepa_results, {})


class TestExportPrompt:
    """Test prompt export functionality."""

    def test_export_prompt_success(self, tmp_path):
        """Test successful export of a prompt."""
        field_data = {
            "field_name": "property_name",
            "type": "string",
            "instructions": "Extract the property name",
            "score": 0.95,
        }

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output_file = export_prompt(field_data, output_dir)

        assert output_file.exists()
        assert output_file.name == "property_name.json"

        with open(output_file) as f:
            saved_data = json.load(f)

        assert saved_data["field_name"] == "property_name"
        assert saved_data["score"] == 0.95


class TestProcessOptimizationLogs:
    """Test the main processing function."""

    def test_process_optimization_logs_success(self, tmp_path):
        """Test successful processing of optimization logs."""
        # Setup directory structure
        input_dir = tmp_path / "optimizer_logs"
        logs_dir = input_dir / "logs"
        logs_dir.mkdir(parents=True)

        # Create config
        config_data = {
            "fields": {
                "field1": {"type": "string", "matcher": "StringMatcher", "json_ref": "F1"},
                "field2": {"type": "date", "matcher": "DateMatcher", "json_ref": "F2"},
            }
        }
        config_path = tmp_path / "fields_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Create GEPA results
        for field_name in ["field1", "field2"]:
            field_dir = logs_dir / field_name
            field_dir.mkdir()
            gepa = {
                "best_idx": 0,
                "candidate_instructions": [
                    {"index": 0, "score": 0.92, "instructions": f"Extract {field_name}", "instruction_length": 50}
                ],
            }
            with open(field_dir / f"gepa_results_{field_name}.json", "w") as f:
                json.dump(gepa, f)

        output_dir = tmp_path / "output"

        # Process
        stats = process_optimization_logs(input_dir, config_path, output_dir)

        # Verify
        assert stats["successful"] == 2
        assert stats["failed"] == 0
        assert stats["skipped"] == 0
        assert (output_dir / "field1.json").exists()
        assert (output_dir / "field2.json").exists()

    def test_process_optimization_logs_with_failures(self, tmp_path):
        """Test processing with missing and failed fields."""
        input_dir = tmp_path / "optimizer_logs"
        logs_dir = input_dir / "logs"
        logs_dir.mkdir(parents=True)

        # Config with 3 fields
        config_data = {
            "fields": {
                "field1": {"type": "string"},
                "field2": {"type": "string"},
                "field3": {"type": "string"},
            }
        }
        config_path = tmp_path / "fields_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Only create GEPA for field1
        field1_dir = logs_dir / "field1"
        field1_dir.mkdir()
        with open(field1_dir / "gepa_results_field1.json", "w") as f:
            json.dump(
                {"best_idx": 0, "candidate_instructions": [{"index": 0, "score": 0.9, "instructions": "Test"}]}, f
            )

        # Create invalid GEPA for field3
        field3_dir = logs_dir / "field3"
        field3_dir.mkdir()
        with open(field3_dir / "gepa_results_field3.json", "w") as f:
            json.dump({"candidate_instructions": []}, f)  # No best_idx

        output_dir = tmp_path / "output"
        stats = process_optimization_logs(input_dir, config_path, output_dir)

        assert stats["successful"] == 1
        assert stats["skipped"] == 1  # field2 missing
        assert stats["failed"] == 1  # field3 invalid

    def test_process_optimization_logs_missing_logs_dir(self, tmp_path):
        """Test processing when logs directory is missing."""
        input_dir = tmp_path / "optimizer_logs"
        input_dir.mkdir()

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"fields": {}}, f)

        with pytest.raises(FileNotFoundError, match="Logs directory not found"):
            process_optimization_logs(input_dir, config_path, tmp_path / "output")
