"""Lightweight evaluation harness — declarative + regression-tested."""

from .critic import Critic
from .eval_spec import EvalSpec, load_eval_specs
from .pairwise import PairwiseComparator
from .regression import GoldenSet, run_regression

__all__ = [
    "EvalSpec",
    "load_eval_specs",
    "Critic",
    "PairwiseComparator",
    "GoldenSet",
    "run_regression",
]
