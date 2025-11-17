"""Shared constants for field extraction matchers."""

# Common null/empty value identifiers
# Used across all matchers to identify values that should be treated as None/null
NULL_VALUES = {
    "",  # Empty string
    "null",
    "none",
    "na",
    "n/a",
    "unknown",
    "missing",
    "tbd",
    "tba",
    "unk",
    "pending",
    "to be determined",
    "0",  # Sometimes used as a null indicator
}
