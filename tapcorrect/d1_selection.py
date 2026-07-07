from __future__ import annotations

from dataclasses import dataclass


PRIMARY_TIE_DELTA = 0.02
COMPLEMENTARITY_OK_MAX = 0.70
COMPLEMENTARITY_HIGH_MIN = 0.90

SIMPLICITY_ORDER = {
    "harvestability": ("axis", "margin", "expectation", "calibrated_softmax"),
    "uncertainty": (
        "top2_margin",
        "entropy",
        "dist_to_threshold",
        "prototype_disagreement",
        "bootstrap_variance",
    ),
}


@dataclass(frozen=True)
class SelectionResult:
    selected: str
    primary_tied: list[str]
    reason: str


def select_candidate(rows, kind: str, primary_tie_delta=PRIMARY_TIE_DELTA) -> SelectionResult:
    if kind not in ("harvestability", "uncertainty"):
        raise ValueError(f"unknown candidate kind: {kind}")
    if not rows:
        raise ValueError("rows must not be empty")

    values = [(row["candidate"], float(row["primary_value"]), float(row["aux_value"])) for row in rows]
    if kind == "harvestability":
        best_primary = max(primary for _, primary, _ in values)
        primary_tied = [
            name for name, primary, _ in values
            if best_primary - primary < primary_tie_delta
        ]
    else:
        best_primary = min(primary for _, primary, _ in values)
        primary_tied = [
            name for name, primary, _ in values
            if primary - best_primary < primary_tie_delta
        ]

    tied_rows = [item for item in values if item[0] in primary_tied]
    if len(tied_rows) == 1:
        return SelectionResult(selected=tied_rows[0][0], primary_tied=primary_tied, reason="primary")

    best_aux = max(aux for _, _, aux in tied_rows)
    aux_tied = [name for name, _, aux in tied_rows if aux == best_aux]
    if len(aux_tied) == 1:
        return SelectionResult(selected=aux_tied[0], primary_tied=primary_tied, reason="auxiliary")

    order = SIMPLICITY_ORDER[kind]
    selected = min(aux_tied, key=lambda name: order.index(name) if name in order else len(order))
    return SelectionResult(selected=selected, primary_tied=primary_tied, reason="simplicity")


def complementarity_status(correlation: float) -> str:
    abs_corr = abs(float(correlation))
    if abs_corr < COMPLEMENTARITY_OK_MAX:
        return "ok"
    if abs_corr < COMPLEMENTARITY_HIGH_MIN:
        return "review"
    return "high_redundancy"
