"""Expose functions outside module."""

from .dataset_utils.compute_embeddings import compute_embeddings
from .dataset_utils.parsing import parse_embeddings_dataset
from .baseline import baseline_cosine

__all__ = ('compute_embeddings', 'parse_embeddings_dataset', 'baseline_cosine')
