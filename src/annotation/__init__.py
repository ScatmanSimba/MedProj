"""Annotation framework for medication response data.

This package provides tools for creating and managing annotation tasks
for medication response data, including gold set creation and annotation
interfaces.
"""

from .gold_set import create_gold_set, get_preliminary_scores
from .interface import display_post, initialize_session_state, main
from .export import export_to_training_format, load_annotations, aggregate_annotations

__all__ = [
    'create_gold_set',
    'get_preliminary_scores',
    'display_post',
    'initialize_session_state',
    'main',
    'export_to_training_format',
    'load_annotations',
    'aggregate_annotations'
] 