"""Output builder for creating structured JSON from LLM responses."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OutputBuilder:
    """Builder for creating structured JSON output from field extractions."""

    def __init__(self):
        """Initialize the output builder."""
        pass

    def parse_json_ref(self, json_ref: str) -> dict[str, Any]:
        """Parse a json_ref string to extract structure information.

        Args:
            json_ref: Reference string (e.g., "STATIC::Gen Info 1::Gen Info 1|Property Information|Property Name")

        Returns:
            Dictionary with parsed components:
                - type: STATIC, TABLE, or TABLE_FILTER
                - section: The section name
                - path: List of nested keys
        """
        if not json_ref or json_ref == "MISSING":
            return {"type": "UNKNOWN", "section": None, "path": []}

        try:
            parts = json_ref.split("::")

            if len(parts) < 2:
                logger.warning(f"Invalid json_ref format: {json_ref}")
                return {"type": "UNKNOWN", "section": None, "path": []}

            ref_type = parts[0]  # STATIC, TABLE, TABLE_FILTER
            section = parts[1] if len(parts) > 1 else None

            # Parse the remaining path
            if len(parts) > 2:
                # The path contains pipe-separated keys
                path_str = parts[2]
                path = path_str.split("|")
            else:
                path = []

            return {
                "type": ref_type,
                "section": section,
                "path": path,
            }

        except Exception as e:
            logger.error(f"Error parsing json_ref '{json_ref}': {e}")
            return {"type": "UNKNOWN", "section": None, "path": []}

    def set_nested_value(
        self,
        data: dict[str, Any],
        path: list[str],
        value: Any,
    ) -> None:
        """Set a value in a nested dictionary structure.

        Args:
            data: The dictionary to modify
            path: List of keys representing the nested path
            value: The value to set
        """
        if not path:
            return

        # Navigate to the nested location
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Overwrite non-dict values with dict
                logger.warning(f"Overwriting non-dict value at key '{key}' in path {path}")
                current[key] = {}
            current = current[key]

        # Set the final value
        current[path[-1]] = value

    def build_output_structure(
        self,
        field_responses: dict[str, str | None],
        prompts_data: dict[str, dict],
    ) -> dict[str, Any]:
        """Build structured JSON output from field responses.

        Args:
            field_responses: Dictionary mapping field names to LLM responses
            prompts_data: Dictionary mapping field names to prompt metadata

        Returns:
            Structured JSON matching the expected output format
        """
        output = {}

        for field_name, response in field_responses.items():
            if response is None:
                logger.debug(f"Skipping field {field_name}: no response")
                continue

            # Get the prompt metadata for this field
            prompt_data = prompts_data.get(field_name)
            if not prompt_data:
                logger.warning(f"No prompt data for field {field_name}, skipping")
                continue

            # Parse the json_ref to determine where to place this value
            json_ref = prompt_data.get("json_ref", "")
            ref_info = self.parse_json_ref(json_ref)

            if ref_info["type"] == "STATIC":
                # Static field - place in nested structure
                self._add_static_field(output, ref_info, field_name, response)

            elif ref_info["type"] == "TABLE":
                # Table field - response should be JSON array
                self._add_table_field(output, ref_info, field_name, response)

            elif ref_info["type"] == "TABLE_FILTER":
                # Filtered table field - handle as filtered data
                self._add_table_filter_field(output, ref_info, field_name, response)

            else:
                # Unknown type - store in a flat "other_fields" section
                logger.warning(f"Unknown json_ref type for field {field_name}: {ref_info['type']}")
                if "other_fields" not in output:
                    output["other_fields"] = {}
                output["other_fields"][field_name] = response

        return output

    def _add_static_field(
        self,
        output: dict[str, Any],
        ref_info: dict[str, Any],
        field_name: str,
        response: str,
    ) -> None:
        """Add a static field to the output structure.

        Args:
            output: The output dictionary to modify
            ref_info: Parsed json_ref information
            field_name: Name of the field
            response: LLM response value
        """
        section = ref_info["section"]
        path = ref_info["path"]

        if not section:
            logger.warning(f"No section for static field {field_name}")
            return

        # Ensure section exists
        if section not in output:
            output[section] = {}

        # Ensure static_fields exists in section
        if "static_fields" not in output[section]:
            output[section]["static_fields"] = {}

        # Set the value using the path
        if path:
            # Use the last component of the path as the key
            key = "|".join(path)
            output[section]["static_fields"][key] = response
        else:
            output[section]["static_fields"][field_name] = response

    def _add_table_field(
        self,
        output: dict[str, Any],
        ref_info: dict[str, Any],
        field_name: str,
        response: str,
    ) -> None:
        """Add a table field to the output structure.

        Args:
            output: The output dictionary to modify
            ref_info: Parsed json_ref information
            field_name: Name of the field
            response: LLM response (should be JSON array)
        """
        section = ref_info["section"]
        path = ref_info["path"]

        if not section:
            logger.warning(f"No section for table field {field_name}")
            return

        # Try to parse response as JSON
        try:
            table_data = json.loads(response)
            if not isinstance(table_data, list):
                logger.warning(f"Table field {field_name} response is not a list, wrapping")
                table_data = [table_data]
        except json.JSONDecodeError:
            logger.warning(f"Could not parse table field {field_name} as JSON, storing as string")
            table_data = response

        # Ensure section exists
        if section not in output:
            output[section] = {}

        # Ensure tables exists in section
        if "tables" not in output[section]:
            output[section]["tables"] = {}

        # Set the table using the path
        if path:
            key = "|".join(path)
            output[section]["tables"][key] = table_data
        else:
            output[section]["tables"][field_name] = table_data

    def _add_table_filter_field(
        self,
        output: dict[str, Any],
        ref_info: dict[str, Any],
        field_name: str,
        response: str,
    ) -> None:
        """Add a filtered table field to the output structure.

        For now, treats filtered fields similarly to static fields.

        Args:
            output: The output dictionary to modify
            ref_info: Parsed json_ref information
            field_name: Name of the field
            response: LLM response value
        """
        # For TABLE_FILTER, we'll store in static_fields for simplicity
        self._add_static_field(output, ref_info, field_name, response)

    def save_output(
        self,
        output_data: dict[str, Any],
        output_path: Path,
    ) -> bool:
        """Save output data to a JSON file.

        Args:
            output_data: The structured output data
            output_path: Path where to save the JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved output to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving output to {output_path}: {e}")
            return False
