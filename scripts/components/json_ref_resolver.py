from logging import getLogger
from typing import Any

logger = getLogger(__name__)


class JsonRefResolver:
    """JSON reference resolver for field extraction data structures.

    This module provides functionality to resolve field references in the format:
    - STATIC::<section>::<field_key>
    - TABLE::<section>::<table_key>::<column>
    - TABLE_FILTER::<section>::<table_key>::<row_field>::<row_value>::<column>
    """

    @classmethod
    def resolve(cls, data: dict[str, Any], path: str) -> Any | list[Any] | None:
        """Resolve a field reference path to its value(s).

        Args:
            data: Lease data dictionary with section-based structure
            path: Reference path in supported format

        Returns:
            Resolved value(s) or None if not found

        Raises:
            ValueError: If path format is invalid
        """
        if not path or path == "MISSING":
            return None

        try:
            parts = path.split("::")
            if not parts:
                return None

            resolver_type = parts[0]

            if resolver_type == "STATIC":
                return cls._resolve_static(data, parts)
            elif resolver_type == "TABLE":
                return cls._resolve_table(data, parts)
            elif resolver_type == "TABLE_FILTER":
                return cls._resolve_table_filter(data, parts)
            else:
                raise ValueError(f"Unknown path type: {resolver_type}")

        except (IndexError, KeyError) as e:
            logger.error(f"Error resolving path '{path}': {e}")
            return None

    @staticmethod
    def _resolve_static(data: dict[str, Any], parts: list[str]) -> Any | None:
        """Resolve STATIC::<section>::<field_key> path."""
        if len(parts) != 3:
            raise ValueError("STATIC path must have format: STATIC::<section>::<field_key>")

        _, section, field_key = parts
        return data.get(section, {}).get("static_fields", {}).get(field_key)

    @staticmethod
    def _resolve_table(data: dict[str, Any], parts: list[str]) -> list[dict[str, Any]]:
        """Resolve TABLE::<section>::<table_key> path.

        Returns the full table as a list of row dictionaries (JSON format).
        """
        if len(parts) != 3:
            raise ValueError("TABLE path must have format: TABLE::<section>::<table_key>")

        _, section, table_key = parts
        rows = data.get(section, {}).get("tables", {}).get(table_key, [])
        return [row for row in rows if isinstance(row, dict)]

    @staticmethod
    def _resolve_table_filter(data: dict[str, Any], parts: list[str]) -> list[Any]:
        """Resolve TABLE_FILTER::<section>::<table_key>::<row_field>::<row_value>::<column> path."""
        if len(parts) != 6:
            raise ValueError(
                "TABLE_FILTER path must have format: "
                "TABLE_FILTER::<section>::<table_key>::<row_field>::<row_value>::<column>"
            )

        _, section, table_key, row_field, row_value, column = parts
        rows = data.get(section, {}).get("tables", {}).get(table_key, [])

        matched = []
        for row in rows:
            if isinstance(row, dict) and row.get(row_field) == row_value:
                value = row.get(column)
                if value is not None:
                    matched.append(value)

        return matched


def resolve_path(data: dict[str, Any], path: str) -> Any | list[Any] | None:
    """Convenience function for resolving JSON references.

    Args:
        data: Lease data dictionary
        path: Reference path to resolve

    Returns:
        Resolved value(s) or None
    """
    return JsonRefResolver.resolve(data, path)
