"""CPU-only exact budget sweeps over already measured gate/up profiles."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from html import escape
from pathlib import Path

from .allocator import (
    FrontierOverflow,
    InfeasibleBudget,
    Option,
    Unit,
    _finite_cost,
    _integer,
    _validate,
    allocate,
)


def _grid(values, field):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field} must be a nonempty list or tuple")
    for value in values:
        _finite_cost(value, field)
        if value <= 0:
            raise ValueError(f"{field} must contain positive numbers")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")
    return sorted(values)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _profile_units(profile):
    """Parse Option/Unit directly, sharing allocator validation without MLX."""
    if not isinstance(profile, dict) or profile.get("scope") != "coupled_gate_up_pairs":
        raise ValueError("profile.scope must be coupled_gate_up_pairs")
    _integer(profile.get("fixed_parameter_bytes"), "profile.fixed_parameter_bytes")
    _integer(profile.get("uniform4_parameter_bytes"), "profile.uniform4_parameter_bytes", 1)
    if not isinstance(profile.get("units"), list) or not profile["units"]:
        raise ValueError("profile.units must be a nonempty list")
    try:
        units = tuple(
            Unit(unit["name"], tuple(Option(**o) for o in unit["options"]))
            for unit in profile["units"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid profile unit/option schema: {exc}") from exc
    # Reuse the exact allocator's cost and identifier checks, without solving.
    _validate(units)
    for unit in units:
        if not unit.options:
            raise ValueError(f"Unit {unit.name!r} has no options")
        if any(o.backend not in ("stock", "fused") for o in unit.options):
            raise ValueError(f"Unit {unit.name!r} has an unsupported backend")
        baseline = [o for o in unit.options if o.bits == 4 and o.backend == "stock"]
        if len(baseline) != 1:
            raise ValueError(f"Unit {unit.name!r} requires exactly one 4-bit stock option")
    return tuple(sorted((
        Unit(u.name, tuple(sorted(u.options, key=lambda o: o.label))) for u in units
    ), key=lambda u: u.name))


def _solution(plan, fixed):
    return {
        "parameter_bytes": fixed + plan.memory_bytes,
        "gate_up_parameter_bytes": plan.memory_bytes,
        "predicted_gate_up_latency_sum_ms": plan.latency_ms,
        "surrogate_loss": plan.loss,
        "exact": plan.exact,
        "states_considered": plan.states_considered,
        "bits_histogram": dict(sorted(Counter(str(o.bits) for o in plan.choices.values()).items())),
        "backend_histogram": dict(sorted(
            Counter(o.backend for o in plan.choices.values()).items()
        )),
        "choices": {name: asdict(o) for name, o in plan.choices.items()},
    }


def sweep_profile(profile_path, memory_fractions, latency_factors, *, max_states=10000):
    """Return deterministic exact stock/fusion-aware allocations; measure nothing.

    Both strategies use the same uniform4 stock model byte/latency reference.
    Invalid inputs raise ValueError before allocation. Per-row allocator errors
    retain their exception type and message, with feasibility unknown.
    """
    fractions = _grid(memory_fractions, "memory_fractions")
    factors = _grid(latency_factors, "latency_factors")
    _integer(max_states, "max_states", 1)
    path = Path(profile_path)
    raw = path.read_bytes()
    profile = json.loads(raw, object_pairs_hook=_unique_object)
    units = _profile_units(profile)
    stock_units = tuple(Unit(u.name, tuple(o for o in u.options if o.backend == "stock"))
                        for u in units)
    uniform4 = [next(o for o in u.options if o.bits == 4) for u in stock_units]
    fixed = profile["fixed_parameter_bytes"]
    reference_bytes = profile["uniform4_parameter_bytes"]
    if fixed + sum(o.memory_bytes for o in uniform4) != reference_bytes:
        raise ValueError("profile.uniform4_parameter_bytes disagrees with fixed + gate/up bytes")
    reference_latency = sum((o.latency_ms for o in uniform4), 0.0)
    _finite_cost(reference_latency, "uniform4 stock latency sum")
    for factor in factors:
        _finite_cost(reference_latency * factor, "latency_factors scaled budget")
    rows = []
    for fraction in fractions:
        # Exact rational text interpretation avoids binary/Decimal rounding at byte boundaries.
        total_budget = int(Fraction(str(fraction)) * reference_bytes)
        for factor in factors:
            latency_budget = reference_latency * factor
            for strategy, candidates in (("stock", stock_units), ("fusion_aware", units)):
                row = {
                    "memory_fraction": fraction,
                    "latency_factor": factor,
                    "strategy": strategy,
                    "total_parameter_budget_bytes": total_budget,
                    "gate_up_parameter_budget_bytes": total_budget - fixed,
                    "gate_up_latency_budget_ms": latency_budget,
                    "solution": None,
                }
                try:
                    if total_budget < fixed:
                        raise InfeasibleBudget("Total budget is smaller than fixed parameters")
                    plan = allocate(candidates, total_budget - fixed, latency_budget, max_states)
                except (InfeasibleBudget, FrontierOverflow, ValueError, ArithmeticError) as exc:
                    infeasible = isinstance(exc, InfeasibleBudget)
                    kind = ("infeasible_budget" if infeasible else
                            "capped_frontier" if isinstance(exc, FrontierOverflow) else
                            "allocation_error")
                    row.update(
                        status="infeasible" if infeasible else "error",
                        feasible=False if infeasible else None,
                        error={"kind": kind, "type": type(exc).__name__, "message": str(exc)},
                    )
                else:
                    row.update(status="feasible", feasible=True, error=None,
                               solution=_solution(plan, fixed))
                rows.append(row)
    return {
        "schema_version": 1,
        "profile_sha256": hashlib.sha256(raw).hexdigest(),
        "profile_path": str(path.resolve()),
        "scope": profile["scope"],
        "fixed_parameter_bytes": fixed,
        "fixed_parameter_bytes_source": "profile.fixed_parameter_bytes",
        "reference": {
            "strategy": "uniform4_stock",
            "parameter_bytes": reference_bytes,
            "predicted_gate_up_latency_sum_ms": reference_latency,
        },
        "memory_fractions": fractions,
        "latency_factors": factors,
        "max_states": max_states,
        "surrogate_loss_note": (
            "Sum of profiled local gate/up activation relative MSE; "
            "not a quality or perplexity measurement."
        ),
        "latency_note": (
            "Predicted sum of profiled isolated gate/up option latencies; "
            "not measured full-model latency."
        ),
        "parameter_bytes_note": (
            "Resident model parameter bytes including fixed non-gate/up parameters "
            "and profiled packed gate/up quantization metadata; not runtime/KV-cache memory."
        ),
        "rows": rows,
    }


def frontier_svg(result, *, title="Measured-profile allocation sweep"):
    """Standalone SVG scatter of feasible optima, not a measured model frontier.

    No external assets or plotting packages. Fraction controls work when opened
    directly in a browser; static SVG viewers still show all feasible points.
    """
    rows = [r for r in result["rows"] if r["status"] == "feasible"]
    fractions = result["memory_fractions"]
    # Wrap controls so any selected grid size remains inside the SVG viewBox.
    control_lines = (len(fractions) + 6) // 7
    top = 170 + 32 * control_lines
    bottom = top + 390
    height = bottom + 135
    left, right = 110, 1010
    x_values = [r["solution"]["parameter_bytes"] / 1024**2 for r in rows] or [0.0]
    y_values = [r["solution"]["predicted_gate_up_latency_sum_ms"] for r in rows] or [0.0]

    def bounds(values):
        low, high = min(values), max(values)
        padding = max((high - low) * 0.12, abs(high) * 0.01, 0.001)
        return max(0.0, low - padding), high + padding

    xmin, xmax = bounds(x_values)
    ymin, ymax = bounds(y_values)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 {height}" '
        'role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        '<desc id="description">Feasible exact surrogate-loss optima only. '
        f'{escape(result["latency_note"])} {escape(result["surrogate_loss_note"])} '
        f'Profile: {escape(result["profile_path"])}</desc>',
        '<style>text{font-family:system-ui,sans-serif;fill:#243348;font-size:13px}'
        '.point .loss-label{display:none;font-size:12px;paint-order:stroke;'
        'stroke:white;stroke-width:4px;stroke-linejoin:round}'
        '.point:hover .loss-label,.point:focus .loss-label{display:block}'
        '.point:focus{outline:none}.toggle{cursor:pointer}.toggle:focus rect{stroke:#111}'
        '.toggle[aria-pressed="false"]{opacity:.35}</style>',
        f'<rect width="1120" height="{height}" fill="#fff"/>',
        f'<text x="65" y="38" style="font-size:24px;font-weight:650">{escape(title)}</text>',
        '<text x="65" y="64">Exact optima of local activation surrogate; '
        'not measured quality or full-model latency.</text>',
        '<circle cx="72" cy="89" r="5" fill="#2563eb"/>'
        '<text x="85" y="94">stock only</text>',
        '<rect x="217" y="84" width="10" height="10" fill="#c2410c"/>'
        '<text x="234" y="94">fusion-aware</text>',
        '<text x="65" y="120">Toggle memory fractions below. '
        'Hover/focus each point for surrogate loss and selected options.</text>',
    ]
    for index, fraction in enumerate(fractions):
        x, y = 65 + (index % 7) * 144, 132 + (index // 7) * 32
        chunks.extend([
            f'<g class="toggle" role="button" tabindex="0" aria-pressed="true" '
            f'data-select="{index}" aria-label="Toggle memory fraction {escape(str(fraction))}">',
            f'<rect x="{x}" y="{y}" width="130" height="25" rx="5" '
            'fill="#e8edf5" stroke="#ccd5e1"/>',
            f'<text x="{x + 10}" y="{y + 17}">memory {escape(str(fraction))}x</text></g>',
        ])
    for step in range(6):
        x = left + (right - left) * step / 5
        y = bottom - (bottom - top) * step / 5
        xv = xmin + (xmax - xmin) * step / 5
        yv = ymin + (ymax - ymin) * step / 5
        chunks.extend([
            f'<path d="M{x:.3f} {top} V{bottom}" stroke="#e2e8f0"/>',
            f'<path d="M{left} {y:.3f} H{right}" stroke="#e2e8f0"/>',
            f'<text x="{x:.3f}" y="{bottom + 24}" text-anchor="middle">{xv:.4g}</text>',
            f'<text x="{left - 14}" y="{y + 4:.3f}" text-anchor="end">{yv:.4g}</text>',
        ])
    chunks.extend([
        f'<path d="M{left} {top} V{bottom} H{right}" fill="none" stroke="#334155"/>',
        f'<text x="560" y="{bottom + 55}" text-anchor="middle">'
        'Resident model parameters (MiB)</text>',
        f'<text transform="translate(28 {(top + bottom) / 2}) rotate(-90)" '
        'text-anchor="middle">Predicted gate/up latency sum (ms)</text>',
    ])
    for index, row in enumerate(rows):
        solution = row["solution"]
        x = left + (solution["parameter_bytes"] / 1024**2 - xmin) / (xmax - xmin) * (right - left)
        y = bottom - (solution["predicted_gate_up_latency_sum_ms"] - ymin) / (ymax - ymin) * (
            bottom - top)
        loss = solution["surrogate_loss"]
        options = "; ".join(f"{name}: {o['label']}" for name, o in solution["choices"].items())
        tooltip = (
            f"{row['strategy']}; memory fraction={row['memory_fraction']}; "
            f"latency factor={row['latency_factor']}; surrogate loss={loss:.9g}; "
            f"parameters={solution['parameter_bytes']} bytes; "
            f"predicted gate/up latency sum={solution['predicted_gate_up_latency_sum_ms']:.9g} ms; "
            f"{options}"
        )
        color = "#2563eb" if row["strategy"] == "stock" else "#c2410c"
        chunks.append(f'<g class="point" tabindex="0" data-row="{index}" '
                      f'data-fraction="{fractions.index(row["memory_fraction"])}" '
                      f'aria-label="{escape(tooltip)}"><title>{escape(tooltip)}</title>')
        if row["strategy"] == "stock":
            chunks.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5.5" '
                          f'fill="{color}" stroke="white"/>')
        else:
            chunks.append(f'<rect x="{x - 5:.3f}" y="{y - 5:.3f}" width="10" height="10" '
                          f'fill="{color}" stroke="white"/>')
        label_x = x - 12 if x > (left + right) / 2 else x + 12
        anchor = "end" if x > (left + right) / 2 else "start"
        chunks.append(f'<text class="loss-label" x="{label_x:.3f}" y="{y - 10:.3f}" '
                      f'text-anchor="{anchor}">surrogate loss={loss:.6g}</text></g>')
    if not rows:
        chunks.append(f'<text x="560" y="{(top + bottom) / 2}" text-anchor="middle">'
                      'No feasible allocations</text>')
    counts = Counter(r["status"] for r in result["rows"])
    chunks.extend([
        f'<text x="65" y="{bottom + 87}">Rows: {len(result["rows"])}; '
        f'feasible: {counts["feasible"]}; infeasible: {counts["infeasible"]}; '
        f'errors / unknown feasibility: {counts["error"]}. '
        'Repeated optima can overlap; no connecting curve is inferred.</text>',
        f'<text x="65" y="{bottom + 111}" style="font-size:11px">'
        f'Profile SHA256: {escape(result["profile_sha256"])}</text>',
        '<script><![CDATA[\n'
        'document.querySelectorAll(".toggle").forEach(function(button) {\n'
        '  function toggle() {\n'
        '    var show = button.getAttribute("aria-pressed") !== "true";\n'
        '    button.setAttribute("aria-pressed", String(show));\n'
        '    document.querySelectorAll(".point").forEach(function(point) {\n'
        '      if (point.dataset.fraction === button.dataset.select) {\n'
        '        point.style.display = show ? "" : "none";\n'
        '      }\n'
        '    });\n'
        '  }\n'
        '  button.addEventListener("click", toggle);\n'
        '  button.addEventListener("keydown", function(event) {\n'
        '    if (event.key === "Enter" || event.key === " ") {\n'
        '      event.preventDefault(); toggle();\n'
        '    }\n'
        '  });\n'
        '});\n]]></script>',
        '</svg>\n',
    ])
    return "\n".join(chunks)


def write_sweep(result, output):
    """Save deterministic JSON/SVG without overwriting a nonempty directory."""
    output = Path(output)
    if output.is_symlink() or (output.exists() and (
        not output.is_dir() or any(output.iterdir())
    )):
        raise FileExistsError(f"Output must be absent or an empty directory; nonempty: {output}")
    json_text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    svg_text = frontier_svg(result)
    output.mkdir(parents=True, exist_ok=True)
    paths = output / "sweep.json", output / "frontier.svg"
    for path, text in zip(paths, (json_text, svg_text), strict=True):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    return paths
