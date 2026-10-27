"""Typed search-space primitives for AGS v3.2."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, List

import numpy as np


@dataclass(frozen=True)
class Parameter:
    """Base class for one discrete hyperparameter dimension."""

    values: tuple

    def __init__(self, values: Iterable[Any]):
        vals = tuple(values)
        if not vals:
            raise ValueError("Parameter values must contain at least one value.")
        object.__setattr__(self, "values", vals)

    def distance(self, left_index: int, right_index: int) -> float:
        raise NotImplementedError

    def neighbor_indices(self, index: int) -> List[int]:
        raise NotImplementedError

    def encode(self, index: int) -> List[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class NumericParameter(Parameter):
    """
    Ordered numeric parameter with normalized value-space geometry.

    scale="linear" uses normalized absolute numeric distance.
    scale="log" uses normalized absolute log-distance and therefore
    requires strictly positive values.
    """

    scale: str = "linear"

    def __init__(self, values: Iterable[Real], scale: str = "linear"):
        super().__init__(values)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in self.values):
            raise TypeError("NumericParameter values must all be real numbers (excluding bool).")
        if scale not in {"linear", "log"}:
            raise ValueError("NumericParameter scale must be 'linear' or 'log'.")
        if scale == "log" and any(float(v) <= 0 for v in self.values):
            raise ValueError("Log-scaled NumericParameter values must all be > 0.")
        object.__setattr__(self, "scale", scale)

    def _transformed(self) -> np.ndarray:
        arr = np.asarray(self.values, dtype=float)
        return np.log(arr) if self.scale == "log" else arr

    def _normalized_value(self, index: int) -> float:
        arr = self._transformed()
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi == lo:
            return 0.0
        return float((arr[index] - lo) / (hi - lo))

    def distance(self, left_index: int, right_index: int) -> float:
        return abs(self._normalized_value(left_index) - self._normalized_value(right_index))

    def neighbor_indices(self, index: int) -> List[int]:
        neighbors = []
        if index > 0:
            neighbors.append(index - 1)
        if index + 1 < len(self.values):
            neighbors.append(index + 1)
        return neighbors

    def encode(self, index: int) -> List[float]:
        return [self._normalized_value(index)]


@dataclass(frozen=True)
class OrdinalParameter(Parameter):
    """Ordered discrete parameter using normalized rank geometry."""

    def distance(self, left_index: int, right_index: int) -> float:
        denominator = max(len(self.values) - 1, 1)
        return abs(left_index - right_index) / denominator

    def neighbor_indices(self, index: int) -> List[int]:
        neighbors = []
        if index > 0:
            neighbors.append(index - 1)
        if index + 1 < len(self.values):
            neighbors.append(index + 1)
        return neighbors

    def encode(self, index: int) -> List[float]:
        denominator = max(len(self.values) - 1, 1)
        return [index / denominator]


@dataclass(frozen=True)
class CategoricalParameter(Parameter):
    """Nominal parameter using Hamming geometry and one-hot encoding."""

    def distance(self, left_index: int, right_index: int) -> float:
        return 0.0 if left_index == right_index else 1.0

    def neighbor_indices(self, index: int) -> List[int]:
        return [i for i in range(len(self.values)) if i != index]

    def encode(self, index: int) -> List[float]:
        encoded = [0.0] * len(self.values)
        encoded[index] = 1.0
        return encoded


def infer_parameter(spec) -> Parameter:
    """
    Convert a legacy param-grid value sequence into a typed parameter.

    Pure real-number grids are inferred as NumericParameter.
    All other grids are treated as nominal categorical parameters.
    Explicit Parameter objects bypass inference.
    """
    if isinstance(spec, Parameter):
        return spec

    values = tuple(spec)
    if not values:
        raise ValueError("Hyperparameter grids must not contain empty value lists.")

    is_numeric = all(
        isinstance(v, Real) and not isinstance(v, (bool, np.bool_))
        for v in values
    )
    if is_numeric:
        return NumericParameter(values)
    return CategoricalParameter(values)
