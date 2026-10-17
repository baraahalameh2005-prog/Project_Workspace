"""fit_engine — clothing fit evaluation and size recommendation.

Standalone module of the Palestinian virtual try-on graduation project.
Uses only the Python standard library.

    from fit_engine import recommend_size
    result = recommend_size(body_doc, garment_doc)
"""

from .adapters import avatar_to_body
from .engine import ENGINE_VERSION, evaluate_size, recommend_size
from .validation import FitInputError, normalize_body, normalize_garment

__all__ = [
    "recommend_size",
    "avatar_to_body",
    "evaluate_size",
    "normalize_body",
    "normalize_garment",
    "FitInputError",
    "ENGINE_VERSION",
]
__version__ = ENGINE_VERSION
