"""The Report Card threshold slot — champion bands, challenger, and scoring.

alpha-engine-config#7476 (epic #7473). See ``registry.py`` for what the slot is
and ``scoring.py`` for how both arms are graded on predictive validity.
"""

from grading.thresholds.registry import (
    DEFAULT_BAND,
    Band,
    ThresholdRegistry,
    ThresholdRegistryError,
    ThresholdRegistrySchemaError,
    load_registry,
    resolve,
)

__all__ = [
    "DEFAULT_BAND",
    "Band",
    "ThresholdRegistry",
    "ThresholdRegistryError",
    "ThresholdRegistrySchemaError",
    "load_registry",
    "resolve",
]
