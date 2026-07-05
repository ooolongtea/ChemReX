"""Mapping-guided reaction validation and conservative correction (Module 5).

Public API: ``ReactionValidator.validate_record(record)`` for a standardized
NER+Action record. Temp/Time/Yield/Action are never modified.
"""

from .validator import ReactionValidator

__all__ = ["ReactionValidator"]
