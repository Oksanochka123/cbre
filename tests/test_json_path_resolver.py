import pytest

from scripts.components.json_ref_resolver import JsonRefResolver, resolve_path


@pytest.fixture
def sample_lease_data():
    """Sample lease data structure for testing."""
    return {
        "Gen Info 1": {
            "static_fields": {
                "Gen Info 1|Property Information|Property Name": "Community Corporate Center",
                "Gen Info 1|Lease Information|Actual Lease Start Date": "2023-12-01 00:00:00",
                "Gen Info 1|Property Information|Country": "United States",
            },
            "tables": {
                "Gen Info 1|Premise Information": [
                    {
                        "Floor Number": "4",
                        "Unit Number": 400,
                        "Rentable/Leasable SF": 20610,
                        "Move-In Date": "2024-02-01 00:00:00",
                    },
                    {
                        "Floor Number": "5",
                        "Unit Number": 500,
                        "Rentable/Leasable SF": 15000,
                        "Move-In Date": "2024-03-01 00:00:00",
                    },
                ]
            },
        },
        "Opt - Misc": {
            "static_fields": {"Opt - Misc|Insurance|Insurance Required (Y/N)?": "Yes"},
            "tables": {
                "Opt - Misc|Right of First Offer or Refusal": [
                    {
                        "Right of First Offer or Refusal": "Right of First Offer (ROFO)",
                        "In Lease?": "No",
                        "Expiration Date": None,
                    },
                    {
                        "Right of First Offer or Refusal": "Right of First Refusal (ROFR)",
                        "In Lease?": "Yes",
                        "Expiration Date": "2025-12-31",
                    },
                ]
            },
        },
        "oPx - CAM": {
            "static_fields": {"oPx - CAM|Lease Type & Tenant Expense Recapture|Lease Type": "NNN"},
            "tables": {
                "oPx - CAM|Tenant Expense Recapture": [
                    {
                        "Unit Number": "400",
                        "ProRata Share %": 0.084,
                        "Fixed ProRata Share?": "Yes",
                        "Base Year": "2024",
                    }
                ]
            },
        },
    }


class TestJsonRefResolver:
    """Test JSON path resolution functionality."""

    def test_static_field_resolution(self, sample_lease_data):
        """Test resolving STATIC field paths."""
        # Test existing field
        result = JsonRefResolver.resolve(
            sample_lease_data, "STATIC::Gen Info 1::Gen Info 1|Property Information|Property Name"
        )
        assert result == "Community Corporate Center"

        # Test date field
        result = JsonRefResolver.resolve(
            sample_lease_data, "STATIC::Gen Info 1::Gen Info 1|Lease Information|Actual Lease Start Date"
        )
        assert result == "2023-12-01 00:00:00"

        # Test different section
        result = JsonRefResolver.resolve(
            sample_lease_data, "STATIC::Opt - Misc::Opt - Misc|Insurance|Insurance Required (Y/N)?"
        )
        assert result == "Yes"

    def test_table_field_resolution(self, sample_lease_data):
        """Test resolving TABLE field paths."""
        # Test full table resolution - returns JSON (list of dicts)
        result = JsonRefResolver.resolve(sample_lease_data, "TABLE::Gen Info 1::Gen Info 1|Premise Information")
        assert result == [
            {
                "Floor Number": "4",
                "Unit Number": 400,
                "Rentable/Leasable SF": 20610,
                "Move-In Date": "2024-02-01 00:00:00",
            },
            {
                "Floor Number": "5",
                "Unit Number": 500,
                "Rentable/Leasable SF": 15000,
                "Move-In Date": "2024-03-01 00:00:00",
            },
        ]

        # Test another table
        result = JsonRefResolver.resolve(sample_lease_data, "TABLE::oPx - CAM::oPx - CAM|Tenant Expense Recapture")
        assert result == [
            {
                "Unit Number": "400",
                "ProRata Share %": 0.084,
                "Fixed ProRata Share?": "Yes",
                "Base Year": "2024",
            }
        ]

    def test_table_filter_resolution(self, sample_lease_data):
        """Test resolving TABLE_FILTER field paths."""
        # Test filtering by string value
        result = JsonRefResolver.resolve(
            sample_lease_data,
            "TABLE_FILTER::Opt - Misc::Opt - Misc|Right of First Offer or Refusal::"
            "Right of First Offer or Refusal::Right of First Offer (ROFO)::In Lease?",
        )
        assert result == ["No"]

        # Test filtering by different value
        result = JsonRefResolver.resolve(
            sample_lease_data,
            "TABLE_FILTER::Opt - Misc::Opt - Misc|Right of First Offer or Refusal::"
            "Right of First Offer or Refusal::Right of First Refusal (ROFR)::In Lease?",
        )
        assert result == ["Yes"]

        # Test filtering with None value result
        result = JsonRefResolver.resolve(
            sample_lease_data,
            "TABLE_FILTER::Opt - Misc::Opt - Misc|Right of First Offer or Refusal::"
            "Right of First Offer or Refusal::Right of First Offer (ROFO)::Expiration Date",
        )
        assert result == []  # None values are not included

        # Test filtering with non-None value
        result = JsonRefResolver.resolve(
            sample_lease_data,
            "TABLE_FILTER::Opt - Misc::Opt - Misc|Right of First Offer or Refusal::"
            "Right of First Offer or Refusal::Right of First Refusal (ROFR)::Expiration Date",
        )
        assert result == ["2025-12-31"]

    def test_missing_and_invalid_paths(self, sample_lease_data):
        """Test handling of missing and invalid paths."""
        # Test MISSING path
        assert JsonRefResolver.resolve(sample_lease_data, "MISSING") is None
        assert JsonRefResolver.resolve(sample_lease_data, "") is None
        assert JsonRefResolver.resolve(sample_lease_data, None) is None

        # Test non-existent section
        result = JsonRefResolver.resolve(sample_lease_data, "STATIC::Non Existent Section::Some Field")
        assert result is None

        # Test non-existent field
        result = JsonRefResolver.resolve(sample_lease_data, "STATIC::Gen Info 1::Non Existent Field")
        assert result is None

        # Test non-existent table
        result = JsonRefResolver.resolve(sample_lease_data, "TABLE::Gen Info 1::Non Existent Table")
        assert result == []

    def test_invalid_path_formats(self, sample_lease_data):
        """Test error handling for invalid path formats."""
        # Test unknown path type
        with pytest.raises(ValueError, match="Unknown path type"):
            JsonRefResolver.resolve(sample_lease_data, "UNKNOWN::Section::Field")

        # Test invalid STATIC format
        with pytest.raises(ValueError, match="STATIC path must have format"):
            JsonRefResolver.resolve(sample_lease_data, "STATIC::Section")

        # Test invalid TABLE format
        with pytest.raises(ValueError, match="TABLE path must have format"):
            JsonRefResolver.resolve(sample_lease_data, "TABLE::Section")

        # Test invalid TABLE_FILTER format
        with pytest.raises(ValueError, match="TABLE_FILTER path must have format"):
            JsonRefResolver.resolve(sample_lease_data, "TABLE_FILTER::Section::Table::Field")

    def test_convenience_function(self, sample_lease_data):
        """Test the convenience resolve_path function."""
        # Test static resolution
        result = resolve_path(
            sample_lease_data,
            "STATIC::Gen Info 1::Gen Info 1|Property Information|Country",
        )
        assert result == "United States"

        # Test table resolution - returns full table as JSON
        result = resolve_path(
            sample_lease_data,
            "TABLE::Gen Info 1::Gen Info 1|Premise Information",
        )
        assert result == [
            {
                "Floor Number": "4",
                "Unit Number": 400,
                "Rentable/Leasable SF": 20610,
                "Move-In Date": "2024-02-01 00:00:00",
            },
            {
                "Floor Number": "5",
                "Unit Number": 500,
                "Rentable/Leasable SF": 15000,
                "Move-In Date": "2024-03-01 00:00:00",
            },
        ]

        # Test missing path
        result = resolve_path(sample_lease_data, "MISSING")
        assert result is None

    def test_edge_cases(self, sample_lease_data):
        """Test edge cases and boundary conditions."""
        # Test empty tables
        sample_lease_data["Empty Section"] = {
            "static_fields": {},
            "tables": {"Empty Table": []},
        }

        result = JsonRefResolver.resolve(sample_lease_data, "TABLE::Empty Section::Empty Table")
        assert result == []

        # Test table with non-dict entries (should be filtered out)
        sample_lease_data["Mixed Section"] = {
            "static_fields": {},
            "tables": {
                "Mixed Table": [
                    {"Column": "Value1"},
                    "invalid_entry",  # Non-dict entry
                    {"Column": "Value2"},
                    None,  # None entry
                ]
            },
        }

        result = JsonRefResolver.resolve(sample_lease_data, "TABLE::Mixed Section::Mixed Table")
        assert result == [{"Column": "Value1"}, {"Column": "Value2"}]  # Non-dict entries filtered out
