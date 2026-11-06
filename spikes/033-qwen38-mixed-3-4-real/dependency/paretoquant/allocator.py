"""Pure-Python allocation of supplied precision costs; no hardware is measured here."""

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import isfinite


class InfeasibleBudget(ValueError):
    """No complete selection fits the supplied budgets."""


class FrontierOverflow(RuntimeError):
    """The exact Pareto frontier exceeds the configured state limit."""


@dataclass(frozen=True)
class Option:
    label: str
    bits: int
    memory_bytes: int
    latency_ms: float
    loss: float
    backend: str = "stock"


@dataclass(frozen=True)
class Unit:
    name: str
    options: tuple[Option, ...]


@dataclass(frozen=True)
class Plan:
    choices: dict[str, Option]
    memory_bytes: int
    latency_ms: float
    loss: float
    exact: bool
    states_considered: int

    def to_dict(self) -> dict:
        """Return JSON-compatible primitive values, including each option's metadata."""
        return asdict(self)


def _rank(state: Plan) -> tuple:
    return (
        state.loss,
        state.latency_ms,
        state.memory_bytes,
        tuple(option.label for option in state.choices.values()),
    )


def _pareto(
    candidates: list[Plan],
    unit_name: str,
    max_states: int,
    preserve_labels: bool,
) -> list[Plan]:
    """Loss-sorted O(n log n) dominance sweep with float-safe partial ties.

    Adding the same large float can erase a strict loss/latency advantage.
    Until the last unit, equal-memory states therefore dominate only when
    their label prefix also wins. Strictly smaller integer memory remains
    decisive if both floating-point advantages disappear.
    """
    ordered = sorted(candidates, key=_rank)
    latencies = sorted({state.latency_ms for state in ordered})
    # Prefix minimum (memory, labels) over compressed latency ranks.
    minima: list[tuple[int, tuple[str, ...]] | None] = [None] * (len(latencies) + 1)
    frontier = []
    for state in ordered:
        index = bisect_left(latencies, state.latency_ms) + 1
        labels = tuple(option.label for option in state.choices.values())
        current = (state.memory_bytes, labels)
        cursor = index
        dominated = False
        while cursor:
            previous = minima[cursor]
            if previous is not None and (
                previous <= current if preserve_labels else previous[0] <= current[0]
            ):
                dominated = True
                break
            cursor -= cursor & -cursor
        if dominated:
            continue
        frontier.append(state)
        if len(frontier) > max_states:
            raise FrontierOverflow(
                f"Exact frontier at unit {unit_name!r} exceeds max_states={max_states}; "
                "increase the limit or reduce candidate options"
            )
        cursor = index
        while cursor < len(minima):
            previous = minima[cursor]
            if previous is None or current < previous:
                minima[cursor] = current
            cursor += cursor & -cursor
    return frontier


def _integer(value: object, field: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")


def _finite_cost(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite nonnegative number")
    try:
        finite = isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")


def _validate(units: tuple[Unit, ...]) -> None:
    names = set()
    for unit in units:
        _text(unit.name, "Unit.name")
        if unit.name in names:
            raise ValueError(f"Duplicate unit name: {unit.name!r}")
        names.add(unit.name)
        labels = set()
        for option in unit.options:
            _text(option.label, "Option.label")
            _text(option.backend, "Option.backend")
            _integer(option.bits, "Option.bits", 1)
            _integer(option.memory_bytes, "Option.memory_bytes")
            _finite_cost(option.latency_ms, "Option.latency_ms")
            _finite_cost(option.loss, "Option.loss")
            if option.label in labels:
                raise ValueError(f"Duplicate option label {option.label!r} in unit {unit.name!r}")
            labels.add(option.label)


def allocate(
    units: Iterable[Unit],
    memory_budget_bytes: int,
    latency_budget_ms: float | None = None,
    max_states: int = 10000,
) -> Plan:
    _integer(memory_budget_bytes, "memory_budget_bytes")
    _integer(max_states, "max_states", 1)
    if latency_budget_ms is not None:
        _finite_cost(latency_budget_ms, "latency_budget_ms")
    units = tuple(units)
    _validate(units)
    frontier = [Plan({}, 0, 0.0, 0.0, True, 0)]
    considered = 0
    units = tuple(sorted(units, key=lambda unit: unit.name))
    for position, unit in enumerate(units):
        candidates = []
        for state in frontier:
            for option in unit.options:
                considered += 1
                memory = state.memory_bytes + option.memory_bytes
                latency = state.latency_ms + option.latency_ms
                if memory > memory_budget_bytes:
                    continue
                if latency_budget_ms is not None and latency > latency_budget_ms:
                    continue
                loss = state.loss + option.loss
                if not isfinite(latency):
                    raise ValueError(f"Accumulated latency_ms is nonfinite at unit {unit.name!r}")
                if not isfinite(loss):
                    raise ValueError(f"Accumulated loss is nonfinite at unit {unit.name!r}")
                candidates.append(
                    Plan(
                        {**state.choices, unit.name: option},
                        memory,
                        latency,
                        loss,
                        True,
                        considered,
                    )
                )
        if not candidates:
            raise InfeasibleBudget(f"No selection fits the budgets at unit {unit.name!r}")
        frontier = _pareto(candidates, unit.name, max_states, position < len(units) - 1)
    best = min(frontier, key=_rank)
    return Plan(best.choices, best.memory_bytes, best.latency_ms, best.loss, True, considered)
