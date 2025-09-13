"""act_mlp: EEG training utilities (EDF loader, features, models)."""

from .datasets import EdfWindowConfig, load_edf_windows
from .feature_extractor import ActExtractorConfig, ActFeatureExtractor
from .model import MLPConfig, build_mlp

__all__ = [
    "EdfWindowConfig",
    "load_edf_windows",
    "ActExtractorConfig",
    "ActFeatureExtractor",
    "MLPConfig",
    "build_mlp",
]
