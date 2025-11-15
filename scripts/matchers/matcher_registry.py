"""Registry for matcher types."""

from matchers.address_matcher import AddressMatcher
from matchers.base_matcher import BaseMatcher
from matchers.boolean_matcher import BooleanMatcher
from matchers.date_matcher import DateMatcher
from matchers.enum_matcher import EnumMatcher
from matchers.float_matcher import FloatMatcher
from matchers.json_matcher import JSONMatcher
from matchers.number_matcher import NumberMatcher
from matchers.phone_matcher import PhoneMatcher
from matchers.string_matcher import StringMatcher


class MatcherRegistry:
    """Registry for matcher types."""

    _matchers = {
        "phone": PhoneMatcher,
        "date": DateMatcher,
        "number": NumberMatcher,
        "float": FloatMatcher,
        "string": StringMatcher,
        "boolean": BooleanMatcher,
        "boolean_string": BooleanMatcher,
        "enum": EnumMatcher,
        "address": AddressMatcher,
        "json": JSONMatcher,
    }

    @classmethod
    def create(cls, field_name: str, field_type: str, **kwargs) -> BaseMatcher:
        """Create matcher for field type."""
        matcher_cls = cls._matchers.get(field_type.lower(), StringMatcher)
        return matcher_cls(field_name, **kwargs)

    @classmethod
    def register(cls, field_type: str, matcher_cls: type):
        """Register custom matcher."""
        cls._matchers[field_type.lower()] = matcher_cls
