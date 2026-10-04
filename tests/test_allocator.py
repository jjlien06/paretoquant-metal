"""Allocator tests: all costs here are synthetic unit-test fixtures, not measurements."""

import itertools
import json
import random

import pytest

from paretoquant.allocator import (
    FrontierOverflow,
    InfeasibleBudget,
    Option,
    Plan,
    Unit,
    allocate,
)


def test_empty_units_return_exact_zero_plan_and_json():
    plan = allocate([], memory_budget_bytes=0, latency_budget_ms=0)
    assert isinstance(plan, Plan)
    assert plan.choices == {}
    assert (plan.memory_bytes, plan.latency_ms, plan.loss) == (0, 0.0, 0.0)
    assert plan.exact is True
    assert plan.states_considered == 0
    assert json.loads(json.dumps(plan.to_dict())) == {
        "choices": {},
        "memory_bytes": 0,
        "latency_ms": 0.0,
        "loss": 0.0,
        "exact": True,
        "states_considered": 0,
    }


def test_one_unit_uses_supplied_bytes_and_preserves_metadata():
    small = Option("three", 3, 17, 2.0, 0.8)
    padded = Option("four-padded", 4, 101, 0.5, 0.1, "fused")
    unit = Unit("projection", (padded, small))
    plan = allocate([unit], 100)
    assert plan.choices == {"projection": small}
    assert (plan.memory_bytes, plan.latency_ms, plan.loss) == (17, 2.0, 0.8)
    assert plan.states_considered == 2
    assert plan.exact is True
    full = allocate([unit], 101)
    assert full.choices == {"projection": padded}
    serialized = json.loads(json.dumps(full.to_dict()))
    assert serialized["choices"]["projection"] == {
        "label": "four-padded",
        "bits": 4,
        "memory_bytes": 101,
        "latency_ms": 0.5,
        "loss": 0.1,
        "backend": "fused",
    }


def test_impossible_memory_budget_raises():
    with pytest.raises(InfeasibleBudget, match="projection"):
        allocate([Unit("projection", (Option("q3", 3, 17, 2.0, 0.8),))], 16)


def test_dual_budget_retains_tradeoffs_and_coupled_gate_up_is_one_unit():
    pair = Unit(
        "gate+up",
        (
            Option("joint-q3", 3, 10, 4.0, 0.9),
            Option("joint-q6", 6, 20, 1.0, 0.1, "fused"),
        ),
    )
    down = Unit(
        "down",
        (
            Option("q3", 3, 10, 1.0, 0.7),
            Option("q6", 6, 20, 4.0, 0.0),
        ),
    )
    plan = allocate([pair, down], 30, latency_budget_ms=2.0)
    assert plan.choices == {"gate+up": pair.options[1], "down": down.options[0]}
    assert (plan.memory_bytes, plan.latency_ms) == (30, 2.0)
    assert plan.loss == pytest.approx(0.8)
    assert len(plan.choices) == 2  # No independent gate/up decisions are introduced.
    with pytest.raises(InfeasibleBudget):
        allocate([pair, down], 29, latency_budget_ms=2.0)


def test_latency_cap_is_strict_without_epsilon():
    units = [
        Unit("a", (Option("q3", 3, 1, 0.1, 0.0),)),
        Unit("b", (Option("q3", 3, 1, 0.2, 0.0),)),
    ]
    with pytest.raises(InfeasibleBudget):
        allocate(units, 2, latency_budget_ms=0.3)
    assert allocate(units, 2, latency_budget_ms=0.1 + 0.2).latency_ms == 0.1 + 0.2


def test_dominated_partial_states_are_removed_before_next_expansion():
    first = Unit(
        "first",
        (
            Option("best", 3, 1, 1.0, 0.0),
            Option("larger", 4, 2, 1.0, 0.0),
            Option("slower", 6, 1, 2.0, 0.0),
            Option("worse", 3, 1, 1.0, 1.0),
        ),
    )
    second = Unit(
        "second",
        (
            Option("a", 3, 1, 1.0, 0.0),
            Option("z", 4, 1, 1.0, 0.0),
        ),
    )
    plan = allocate([first, second], 100, max_states=1)
    assert plan.choices == {"first": first.options[0], "second": second.options[0]}
    assert plan.states_considered == 6  # Four candidates, then only 1 x 2.


def test_frontier_overflow_raises_instead_of_claiming_approximation_is_exact():
    unit = Unit(
        "tradeoff",
        (
            Option("q3", 3, 1, 0.0, 2.0),
            Option("q4", 4, 2, 0.0, 1.0),
            Option("q6", 6, 3, 0.0, 0.0),
        ),
    )
    with pytest.raises(FrontierOverflow, match="tradeoff.*max_states=2"):
        allocate([unit], 3, max_states=2)
    assert allocate([unit], 3, max_states=3).exact is True


def test_ties_use_loss_latency_memory_then_labels_in_sorted_unit_order():
    options = (
        Option("a", 4, 2, 0.0, 0.0),
        Option("z", 3, 1, 0.0, 1.0),
    )
    units = [Unit("a-unit", options), Unit("b-unit", options)]
    for order in (units, list(reversed(units))):
        plan = allocate(order, 3)
        assert plan.choices == {"a-unit": options[0], "b-unit": options[1]}
    latency_tie = Unit(
        "unit",
        (
            Option("a-slow", 3, 1, 2.0, 0.0),
            Option("z-fast", 6, 2, 1.0, 0.0),
        ),
    )
    assert allocate([latency_tie], 2).choices["unit"] == latency_tie.options[1]
    memory_tie = Unit(
        "unit",
        (
            Option("a-big", 6, 2, 1.0, 0.0),
            Option("z-small", 3, 1, 1.0, 0.0),
        ),
    )
    assert allocate([memory_tie], 2).choices["unit"] == memory_tie.options[1]


def _brute_force(units, memory_budget, latency_budget):
    ordered = sorted(units, key=lambda unit: unit.name)
    candidates = []
    for selection in itertools.product(*(unit.options for unit in ordered)):
        memory = sum(option.memory_bytes for option in selection)
        latency = sum((option.latency_ms for option in selection), 0.0)
        loss = sum((option.loss for option in selection), 0.0)
        if memory > memory_budget or (latency_budget is not None and latency > latency_budget):
            continue
        key = (loss, latency, memory, tuple(option.label for option in selection))
        candidates.append((key, dict(zip((unit.name for unit in ordered), selection))))
    return min(candidates, key=lambda candidate: candidate[0]) if candidates else None


@pytest.mark.parametrize("seed", range(12))
def test_seeded_small_candidates_match_brute_force(seed):
    rng = random.Random(seed)
    for _ in range(25):
        units = [
            Unit(
                f"unit-{i}",
                tuple(
                    Option(
                        f"option-{j}",
                        rng.choice((3, 4, 6)),
                        rng.randrange(10),
                        rng.randrange(10) / 4.0,
                        rng.randrange(10) / 4.0,
                    )
                    for j in range(rng.randrange(1, 5))
                ),
            )
            for i in range(rng.randrange(1, 6))
        ]
        rng.shuffle(units)
        memory = rng.randrange(35)
        latency = rng.choice((None, rng.randrange(35) / 4.0))
        expected = _brute_force(units, memory, latency)
        if expected is None:
            with pytest.raises(InfeasibleBudget):
                allocate(units, memory, latency)
        else:
            key, choices = expected
            plan = allocate(units, memory, latency)
            assert plan.choices == choices
            assert (plan.loss, plan.latency_ms, plan.memory_bytes) == key[:3]
            assert plan.exact is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory_budget_bytes": -1},
        {"memory_budget_bytes": 1.5},
        {"memory_budget_bytes": float("nan")},
        {"memory_budget_bytes": float("inf")},
        {"memory_budget_bytes": True},
        {"memory_budget_bytes": "10"},
        {"latency_budget_ms": -1.0},
        {"latency_budget_ms": float("nan")},
        {"latency_budget_ms": float("inf")},
        {"latency_budget_ms": True},
        {"latency_budget_ms": "10"},
        {"max_states": 0},
        {"max_states": -1},
        {"max_states": 1.5},
        {"max_states": True},
    ],
)
def test_invalid_budgets_or_frontier_limit_raise_even_for_empty_units(kwargs):
    arguments = {"memory_budget_bytes": 10, **kwargs}
    with pytest.raises(ValueError):
        allocate([], **arguments)


@pytest.mark.parametrize(
    "field,value",
    [
        ("memory_bytes", -1),
        ("memory_bytes", 1.5),
        ("memory_bytes", float("inf")),
        ("memory_bytes", float("nan")),
        ("memory_bytes", True),
        ("latency_ms", -1.0),
        ("latency_ms", float("inf")),
        ("latency_ms", float("nan")),
        ("latency_ms", "1.0"),
        ("latency_ms", True),
        ("loss", -1.0),
        ("loss", float("inf")),
        ("loss", float("nan")),
        ("loss", "0.1"),
        ("loss", True),
        ("bits", 0),
        ("bits", -1),
        ("bits", 3.5),
        ("bits", True),
        ("label", ""),
        ("label", 3),
        ("backend", ""),
        ("backend", None),
    ],
)
def test_invalid_option_metadata_is_rejected(field, value):
    arguments = dict(label="q3", bits=3, memory_bytes=1, latency_ms=1.0, loss=0.0)
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        allocate([Unit("unit", (Option(**arguments),))], 10)


@pytest.mark.parametrize("name", ["", None, 7])
def test_invalid_unit_names_are_rejected(name):
    with pytest.raises(ValueError, match="name"):
        allocate([Unit(name, (Option("q3", 3, 1, 0.0, 0.0),))], 10)


def test_duplicate_unit_names_are_rejected():
    unit = Unit("duplicate", (Option("q3", 3, 1, 0.0, 0.0),))
    with pytest.raises(ValueError, match="Duplicate unit name"):
        allocate([unit, unit], 10)


def test_duplicate_option_labels_are_rejected_within_unit_only():
    options = (Option("duplicate", 3, 1, 0.0, 1.0), Option("duplicate", 6, 2, 0.0, 0.0))
    with pytest.raises(ValueError, match="Duplicate option label"):
        allocate([Unit("unit", options)], 10)
    assert len(allocate([Unit("a", options[:1]), Unit("b", options[:1])], 10).choices) == 2


def test_validation_precedes_search_even_if_first_unit_cannot_fit():
    units = [
        Unit("a", (Option("q3", 3, 100, 0.0, 0.0),)),
        Unit("z", (Option("invalid", 3, 1, float("nan"), 0.0),)),
    ]
    with pytest.raises(ValueError, match="latency_ms"):
        allocate(units, 0)


def test_unit_without_options_is_infeasible():
    with pytest.raises(InfeasibleBudget, match="empty"):
        allocate([Unit("empty", ())], 10)


def test_solver_does_not_restrict_valid_positive_precision_bits():
    option = Option("other-backend", 8, 3, 0.0, 0.0, "external")
    assert allocate([Unit("unit", (option,))], 3).choices["unit"] is option


@pytest.mark.parametrize("field", ["latency_ms", "loss"])
def test_nonfinite_accumulated_cost_is_rejected(field):
    costs = {"latency_ms": 0.0, "loss": 0.0, field: 1e308}
    units = [Unit(name, (Option("q3", 3, 0, **costs),)) for name in ("a", "b")]
    with pytest.raises(ValueError, match=f"Accumulated {field}"):
        allocate(units, 0)


@pytest.mark.parametrize("field", ["latency_ms", "loss"])
def test_pruning_preserves_labels_when_future_float_addition_erases_advantage(field):
    better = {"latency_ms": 0.0, "loss": 0.0}
    worse = {**better, field: 1.0}
    tail = {**better, field: 1e20}
    units = [
        Unit("a-prefix", (Option("z", 3, 1, **better), Option("a", 3, 1, **worse))),
        Unit("z-tail", (Option("tail", 3, 0, **tail),)),
    ]
    expected = _brute_force(units, 1, None)
    plan = allocate(units, 1)
    assert plan.choices == expected[1]
    assert plan.choices["a-prefix"].label == "a"
