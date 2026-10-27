"""Typed, conditional, and lazy search-space primitives for AGS v3.7."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np


INACTIVE_INDEX = -1


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

    @property
    def feature_width(self) -> int:
        return len(self.encode(0))


@dataclass(frozen=True)
class NumericParameter(Parameter):
    """Ordered numeric parameter with normalized linear or log geometry."""

    scale: str = "linear"

    def __init__(self, values: Iterable[Real], scale: str = "linear"):
        super().__init__(values)
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
            for v in self.values
        ):
            raise TypeError(
                "NumericParameter values must all be real numbers (excluding bool)."
            )
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
        return abs(
            self._normalized_value(left_index)
            - self._normalized_value(right_index)
        )

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
class FloatParameter(NumericParameter):
    """Explicit floating-point parameter type."""

    def __init__(self, values: Iterable[Real], scale: str = "linear"):
        vals = tuple(values)
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
            for v in vals
        ):
            raise TypeError("FloatParameter values must all be real numbers.")
        NumericParameter.__init__(self, vals, scale=scale)


@dataclass(frozen=True)
class IntegerParameter(NumericParameter):
    """Explicit integer parameter type."""

    def __init__(self, values: Iterable[Integral], scale: str = "linear"):
        vals = tuple(values)
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
            for v in vals
        ):
            raise TypeError("IntegerParameter values must all be integers.")
        NumericParameter.__init__(self, vals, scale=scale)


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


class ConditionalParameter(Parameter):
    """A parameter active only for selected values of an earlier parent.

    Parameters
    ----------
    parameter:
        A Parameter instance or a legacy iterable of values.
    parent:
        Name of another parameter that appears earlier in the param_grid.
    active_values:
        Parent values for which this parameter is active.

    Inactive dimensions are represented internally by state index -1, are
    omitted from estimator parameters, and receive an explicit active-mask
    feature in surrogate encoding.
    """

    def __init__(
        self,
        parameter: Parameter | Iterable[Any],
        *,
        parent: str,
        active_values: Iterable[Any],
    ):
        base = parameter if isinstance(parameter, Parameter) else infer_parameter(parameter)
        Parameter.__init__(self, base.values)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "parent", str(parent))
        active = tuple(active_values)
        if not active:
            raise ValueError("ConditionalParameter active_values must not be empty.")
        object.__setattr__(self, "active_values", active)

    def distance(self, left_index: int, right_index: int) -> float:
        return self.base.distance(left_index, right_index)

    def neighbor_indices(self, index: int) -> List[int]:
        return self.base.neighbor_indices(index)

    def encode(self, index: int) -> List[float]:
        return self.base.encode(index)

    @property
    def feature_width(self) -> int:
        return 1 + self.base.feature_width


def infer_parameter(spec) -> Parameter:
    """Convert a legacy sequence into a typed parameter."""
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


class StructuredSearchSpace:
    """Finite structured search space with lazy state generation."""

    def __init__(self, param_grid: Mapping[str, Any]):
        if not param_grid:
            raise ValueError("param_grid must contain at least one parameter.")

        self.parameter_names = list(param_grid.keys())
        self.parameters = [infer_parameter(param_grid[name]) for name in self.parameter_names]
        self.values = [list(parameter.values) for parameter in self.parameters]
        self.name_to_index = {
            name: i for i, name in enumerate(self.parameter_names)
        }

        for i, parameter in enumerate(self.parameters):
            if isinstance(parameter, ConditionalParameter):
                if parameter.parent not in self.name_to_index:
                    raise ValueError(
                        f"Conditional parent {parameter.parent!r} is not in param_grid."
                    )
                parent_index = self.name_to_index[parameter.parent]
                if parent_index >= i:
                    raise ValueError(
                        "Conditional parents must appear before their children "
                        "in param_grid insertion order."
                    )
                parent_values = self.parameters[parent_index].values
                missing = [
                    value for value in parameter.active_values
                    if value not in parent_values
                ]
                if missing:
                    raise ValueError(
                        f"Conditional active_values {missing!r} are not values "
                        f"of parent {parameter.parent!r}."
                    )

        upper = 1
        for parameter in self.parameters:
            upper *= len(parameter.values)
        self.size_upper_bound = int(upper)
        self.has_conditionals = any(
            isinstance(parameter, ConditionalParameter)
            for parameter in self.parameters
        )

    def _active_from_partial(
        self,
        parameter: Parameter,
        partial_params: Dict[str, Any],
    ) -> bool:
        if not isinstance(parameter, ConditionalParameter):
            return True
        return partial_params.get(parameter.parent) in parameter.active_values

    def state_to_params(self, state: Sequence[int]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, parameter, index in zip(
            self.parameter_names, self.parameters, state
        ):
            active = self._active_from_partial(parameter, params)
            if not active:
                continue
            if index == INACTIVE_INDEX:
                raise ValueError(
                    f"State marks active parameter {name!r} as inactive."
                )
            params[name] = parameter.values[index]
        return params

    def canonicalize(self, state: Sequence[int]) -> tuple:
        state = list(state)
        params: Dict[str, Any] = {}
        for i, (name, parameter) in enumerate(
            zip(self.parameter_names, self.parameters)
        ):
            active = self._active_from_partial(parameter, params)
            if not active:
                state[i] = INACTIVE_INDEX
                continue
            index = state[i]
            if index == INACTIVE_INDEX:
                index = 0
                state[i] = 0
            if not 0 <= int(index) < len(parameter.values):
                raise ValueError(f"Invalid state index for parameter {name!r}.")
            params[name] = parameter.values[int(index)]
        return tuple(int(index) for index in state)

    def is_valid(self, state: Sequence[int]) -> bool:
        try:
            return tuple(state) == self.canonicalize(state)
        except (ValueError, IndexError, TypeError):
            return False

    def active_mask(self, state: Sequence[int]) -> List[bool]:
        canonical = self.canonicalize(state)
        params: Dict[str, Any] = {}
        mask = []
        for name, parameter, index in zip(
            self.parameter_names, self.parameters, canonical
        ):
            active = self._active_from_partial(parameter, params)
            mask.append(active)
            if active:
                params[name] = parameter.values[index]
        return mask

    def encode(self, state: Sequence[int]) -> List[float]:
        state = self.canonicalize(state)
        mask = self.active_mask(state)
        features: List[float] = []
        for parameter, index, active in zip(
            self.parameters, state, mask
        ):
            if isinstance(parameter, ConditionalParameter):
                features.append(1.0 if active else 0.0)
                if active:
                    features.extend(parameter.base.encode(index))
                else:
                    features.extend([0.0] * parameter.base.feature_width)
            else:
                features.extend(parameter.encode(index))
        return features

    def distance(self, left: Sequence[int], right: Sequence[int]) -> float:
        left = self.canonicalize(left)
        right = self.canonicalize(right)
        left_mask = self.active_mask(left)
        right_mask = self.active_mask(right)
        total = 0.0
        for parameter, li, ri, la, ra in zip(
            self.parameters, left, right, left_mask, right_mask
        ):
            if isinstance(parameter, ConditionalParameter):
                if la != ra:
                    total += 1.0
                elif la and ra:
                    total += parameter.base.distance(li, ri)
            else:
                total += parameter.distance(li, ri)
        return float(total)

    def neighbors(self, state: Sequence[int]) -> List[tuple]:
        state = self.canonicalize(state)
        mask = self.active_mask(state)
        neighbors = set()
        for i, (parameter, active) in enumerate(zip(self.parameters, mask)):
            if not active:
                continue
            base = parameter.base if isinstance(parameter, ConditionalParameter) else parameter
            for new_index in base.neighbor_indices(state[i]):
                candidate = list(state)
                candidate[i] = new_index
                neighbors.add(self.canonicalize(candidate))
        neighbors.discard(tuple(state))
        return list(neighbors)

    def random_state(self, rng: np.random.Generator) -> tuple:
        state = []
        partial_params: Dict[str, Any] = {}
        for name, parameter in zip(self.parameter_names, self.parameters):
            active = self._active_from_partial(parameter, partial_params)
            if not active:
                state.append(INACTIVE_INDEX)
                continue
            index = int(rng.integers(len(parameter.values)))
            state.append(index)
            partial_params[name] = parameter.values[index]
        return tuple(state)

    def iter_states(self) -> Iterator[tuple]:
        """Yield valid states recursively without materializing the space."""

        def walk(i: int, state: List[int], params: Dict[str, Any]):
            if i == len(self.parameters):
                yield tuple(state)
                return

            name = self.parameter_names[i]
            parameter = self.parameters[i]
            active = self._active_from_partial(parameter, params)
            if not active:
                state.append(INACTIVE_INDEX)
                yield from walk(i + 1, state, params)
                state.pop()
                return

            for index, value in enumerate(parameter.values):
                state.append(index)
                params[name] = value
                yield from walk(i + 1, state, params)
                params.pop(name, None)
                state.pop()

        yield from walk(0, [], {})

    def sample_states(
        self,
        rng: np.random.Generator,
        n: int,
        exclude: Iterable[tuple] = (),
    ) -> List[tuple]:
        """Sample up to n unique valid states without building the full grid."""
        n = max(int(n), 0)
        if n == 0:
            return []

        excluded = set(exclude)
        chosen = []
        chosen_set = set()
        max_attempts = max(100, n * 30)

        for _ in range(max_attempts):
            if len(chosen) >= n:
                break
            state = self.random_state(rng)
            if state in excluded or state in chosen_set:
                continue
            chosen.append(state)
            chosen_set.add(state)

        if len(chosen) < n:
            # Deterministic fallback is bounded by n: the generator itself is lazy.
            for state in self.iter_states():
                if state in excluded or state in chosen_set:
                    continue
                chosen.append(state)
                chosen_set.add(state)
                if len(chosen) >= n:
                    break

        return chosen
