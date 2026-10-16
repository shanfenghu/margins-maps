#!/usr/bin/env python3
"""Reproduce the Connecticut county--watershed reaggregation experiment."""

from __future__ import annotations

import copy
import json
import gzip
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon
from pyproj import Transformer
from scipy.optimize import linprog
from shapely.geometry import shape
from shapely.ops import transform


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "usgs"
RAW_AG = ROOT / "data" / "raw" / "usda"
OUT = ROOT / "results"
PLOTS = ROOT / "plots"
MPL = ROOT / ".cache" / "matplotlib"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
MPL.mkdir(parents=True, exist_ok=True)


def read_features(path: Path, key: str) -> dict[str, object]:
    data = json.loads(path.read_text())
    project = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform
    return {
        str(feature["properties"][key]): transform(project, shape(feature["geometry"]))
        for feature in data["features"]
    }


def read_gzip_features(path: Path, key: str) -> dict[str, object]:
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    project = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform
    return {
        str(feature["properties"][key]): transform(project, shape(feature["geometry"]))
        for feature in data["features"]
    }


def build_support(min_area_km2: float = 0.50) -> tuple[list[str], list[str], list[tuple[int, int]], np.ndarray, dict]:
    counties = read_features(RAW / "ct_counties_2010.geojson", "COUNTY")
    current_hucs = read_features(RAW / "ct_huc8.geojson", "huc8")
    water = pd.read_csv(RAW / "ct85w8.txt", sep="\t", dtype={"area": str})
    huc_codes = water["area"].astype(str).str.zfill(8).tolist()
    # The 1985 release used 01090005 for Pawcatuck--Wood; the current WBD
    # names the Connecticut Pawcatuck unit 01100007. The source page records
    # that HUC identifiers have changed. No other released code needs a map.
    geometry_code = {h: ("01100007" if h == "01090005" else h) for h in huc_codes}
    county_codes = sorted(counties)
    edges: list[tuple[int, int]] = []
    areas: list[float] = []
    geometries: dict[tuple[int, int], object] = {}
    for i, county in enumerate(county_codes):
        for j, huc in enumerate(huc_codes):
            geom = counties[county].intersection(current_hucs[geometry_code[huc]])
            area_km2 = geom.area / 1_000_000
            if area_km2 > 0 and area_km2 >= min_area_km2:
                edges.append((i, j))
                areas.append(area_km2)
                geometries[(i, j)] = geom
    return county_codes, huc_codes, edges, np.asarray(areas), geometries


def incidence(na: int, nb: int, edges: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((na + nb, len(edges)))
    for k, (i, j) in enumerate(edges):
        matrix[i, k] = 1.0
        matrix[na + j, k] = 1.0
    return matrix


def incidence_apply(na: int, nb: int, edges: list[tuple[int, int]],
                    values: np.ndarray) -> np.ndarray:
    """Apply the unsigned incidence matrix without a dense BLAS call."""
    result = np.zeros(na + nb, dtype=float)
    for value, (i, j) in zip(values, edges):
        result[i] += value
        result[na + j] += value
    return result


def transport_min_cost_integer(row: np.ndarray, col: np.ndarray,
                               edges: list[tuple[int, int]],
                               costs: np.ndarray) -> int | None:
    """Solve an integer transportation problem without an LP solver.

    Successive shortest augmenting paths use a residual graph and Bellman--Ford
    because maximization introduces negative edge costs.  The implementation
    shares neither the matrix representation nor HiGHS with the primary path.
    """
    row_int = np.rint(row).astype(np.int64)
    col_int = np.rint(col).astype(np.int64)
    cost_int = np.rint(costs).astype(np.int64)
    if (np.max(np.abs(row - row_int)) > 1e-8 or
            np.max(np.abs(col - col_int)) > 1e-8 or
            np.max(np.abs(costs - cost_int)) > 1e-8 or
            np.any(row_int < 0) or np.any(col_int < 0) or
            int(row_int.sum()) != int(col_int.sum())):
        raise ValueError("integer transportation oracle received invalid data")

    na, nb = len(row_int), len(col_int)
    source, row_offset, col_offset = 0, 1, 1 + na
    sink = 1 + na + nb
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_arc(left: int, right: int, capacity: int, cost: int) -> None:
        forward = [right, len(graph[right]), capacity, cost]
        reverse = [left, len(graph[left]), 0, -cost]
        graph[left].append(forward)
        graph[right].append(reverse)

    total = int(row_int.sum())
    for i, supply in enumerate(row_int):
        add_arc(source, row_offset + i, int(supply), 0)
    for (i, j), cost in zip(edges, cost_int):
        add_arc(row_offset + i, col_offset + j, total, int(cost))
    for j, demand in enumerate(col_int):
        add_arc(col_offset + j, sink, int(demand), 0)

    sent = objective = 0
    infinity = 10 ** 30
    while sent < total:
        distance = [infinity] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for left, arcs in enumerate(graph):
                if distance[left] == infinity:
                    continue
                for arc_index, arc in enumerate(arcs):
                    right, _, capacity, cost = arc
                    candidate = distance[left] + cost
                    if capacity > 0 and candidate < distance[right]:
                        distance[right] = candidate
                        previous[right] = (left, arc_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            return None
        amount = total - sent
        node = sink
        while node != source:
            left, arc_index = previous[node]
            amount = min(amount, graph[left][arc_index][2])
            node = left
        node = sink
        while node != source:
            left, arc_index = previous[node]
            arc = graph[left][arc_index]
            right, reverse_index, _, cost = arc
            arc[2] -= amount
            graph[right][reverse_index][2] += amount
            objective += amount * cost
            node = left
        sent += amount
    return objective


def solve_pair(A: np.ndarray, b: np.ndarray, c: np.ndarray,
               upper: np.ndarray | None = None,
               lower: np.ndarray | None = None):
    # One redundant balance row is removed for numerical stability.
    keep = list(range(A.shape[0] - 1))
    lower_values = (np.zeros(A.shape[1]) if lower is None else lower)
    upper_values = ([None] * A.shape[1] if upper is None else upper)
    bounds = [(float(lo), None if hi is None else float(hi))
              for lo, hi in zip(lower_values, upper_values)]
    kwargs = dict(A_eq=A[keep], b_eq=b[keep], bounds=bounds, method="highs")
    return linprog(c, **kwargs), linprog(-c, **kwargs)


def solve_bounds(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                 upper: np.ndarray | None = None,
                 lower: np.ndarray | None = None):
    lo, hi = solve_pair(A, b, c, upper, lower)
    if not lo.success or not hi.success:
        return None
    return float(lo.fun), float(-hi.fun), lo.x, hi.x


def dual_certificate_error(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                           result, lower: np.ndarray | None = None,
                           upper: np.ndarray | None = None) -> float:
    """Replay a HiGHS bound certificate using only KKT equalities."""
    keep = list(range(A.shape[0] - 1))
    lower_values = (np.zeros(A.shape[1]) if lower is None
                    else np.asarray(lower, dtype=float))
    upper_values = (np.full(A.shape[1], np.inf) if upper is None
                    else np.asarray(upper, dtype=float))
    finite_upper = np.isfinite(upper_values)
    y = np.asarray(result.eqlin.marginals)
    low_dual = np.asarray(result.lower.marginals)
    up_dual = np.asarray(result.upper.marginals)
    if not (np.all(np.isfinite(result.x)) and np.all(np.isfinite(y)) and
            np.all(np.isfinite(low_dual)) and np.all(np.isfinite(up_dual))):
        return math.inf
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        stationarity_vector = c - A[keep].T @ y - low_dual - up_dual
    if not np.all(np.isfinite(stationarity_vector)):
        return math.inf
    stationarity = np.max(np.abs(stationarity_vector))
    sign_error = max(float(np.max(np.maximum(-low_dual, 0.0))),
                     float(np.max(np.maximum(up_dual, 0.0))))
    dual_objective = float(b[keep] @ y + lower_values @ low_dual)
    if np.any(finite_upper):
        dual_objective += float(upper_values[finite_upper] @
                                up_dual[finite_upper])
    gap = abs(float(result.fun) - dual_objective)
    lower_complementarity = float(np.max(np.abs(
        low_dual * (result.x - lower_values))))
    upper_complementarity = (float(np.max(np.abs(
        up_dual[finite_upper] *
        (upper_values[finite_upper] - result.x[finite_upper]))))
        if np.any(finite_upper) else 0.0)
    complementarity = max(lower_complementarity, upper_complementarity)
    return max(float(stationarity), sign_error, gap, complementarity)


def endpoint_certificate_error(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                               result, lower: np.ndarray | None = None,
                               upper: np.ndarray | None = None) -> float:
    """Replay primal feasibility, objective binding, and the dual certificate."""
    if not result.success:
        return math.inf
    x = np.asarray(result.x, dtype=float)
    lower_values = (np.zeros(A.shape[1]) if lower is None
                    else np.asarray(lower, dtype=float))
    upper_values = (np.full(A.shape[1], np.inf) if upper is None
                    else np.asarray(upper, dtype=float))
    if not np.all(np.isfinite(x)):
        return math.inf
    finite_upper = np.isfinite(upper_values)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        balance_vector = A @ x - b
        objective_value = c @ x
    if (not np.all(np.isfinite(balance_vector)) or
            not np.isfinite(objective_value)):
        return math.inf
    balance = float(np.max(np.abs(balance_vector)))
    lower_violation = float(np.max(np.maximum(lower_values - x, 0.0)))
    upper_violation = (float(np.max(np.maximum(
        x[finite_upper] - upper_values[finite_upper], 0.0)))
        if np.any(finite_upper) else 0.0)
    objective = abs(float(result.fun) - float(objective_value))
    dual = dual_certificate_error(A, b, c, result, lower, upper)
    return max(balance, lower_violation, upper_violation, objective, dual)


def certificate_mutation_errors(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                                result, lower: np.ndarray | None = None,
                                upper: np.ndarray | None = None) -> dict[str, float]:
    """Challenge every verifier layer with independent certificate corruptions."""
    primal = copy.deepcopy(result)
    primal.x = np.asarray(primal.x, dtype=float).copy()
    primal.x[0] += 1.0

    dual = copy.deepcopy(result)
    dual.eqlin.marginals = np.asarray(dual.eqlin.marginals, dtype=float).copy()
    dual.eqlin.marginals[0] += 1.0

    rebound_c = np.asarray(c, dtype=float).copy()
    rebound_c[0] += 1.0

    objective = copy.deepcopy(result)
    objective.fun = float(objective.fun) + 1.0

    lower_values = (np.zeros(A.shape[1]) if lower is None
                    else np.asarray(lower, dtype=float).copy())
    lower_values[0] = float(result.x[0]) + 1.0

    upper_values = (np.full(A.shape[1], np.inf) if upper is None
                    else np.asarray(upper, dtype=float).copy())
    upper_values[0] = float(result.x[0]) - 1.0

    nonfinite = copy.deepcopy(result)
    nonfinite.x = np.asarray(nonfinite.x, dtype=float).copy()
    nonfinite.x[0] = np.nan
    return {
        "primal-flow": endpoint_certificate_error(
            A, b, c, primal, lower, upper),
        "dual-potential": endpoint_certificate_error(
            A, b, c, dual, lower, upper),
        "query-binding": endpoint_certificate_error(
            A, b, rebound_c, result, lower, upper),
        "claimed-objective": endpoint_certificate_error(
            A, b, c, objective, lower, upper),
        "lower-contract": endpoint_certificate_error(
            A, b, c, result, lower_values, upper),
        "upper-contract": endpoint_certificate_error(
            A, b, c, result, lower, upper_values),
        "nonfinite-primal": endpoint_certificate_error(
            A, b, c, nonfinite, lower, upper),
    }


def violated_neighbor_cut(row: np.ndarray, col: np.ndarray,
                          edges: list[tuple[int, int]]) -> dict:
    """Return the strongest Hall-type cut for an uncapacitated support."""
    best = {"violation": -math.inf}
    for mask in range(1, 1 << len(row)):
        selected = [i for i in range(len(row)) if mask & (1 << i)]
        neighbors = sorted({j for i, j in edges if i in selected})
        required = float(row[selected].sum())
        available = float(col[neighbors].sum()) if neighbors else 0.0
        violation = required - available
        if violation > best["violation"]:
            best = {"rows": selected, "neighbors": neighbors,
                    "required": required, "available": available,
                    "violation": violation}
    return best


def cycle_exactness_index(A: np.ndarray, b: np.ndarray,
                          upper: np.ndarray | None = None,
                          lower: np.ndarray | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Identify nonfixed edges and a numerical basis of their cycle space."""
    widths = []
    for edge in range(A.shape[1]):
        objective = np.zeros(A.shape[1]); objective[edge] = 1.0
        result = solve_bounds(A, b, objective, upper, lower)
        widths.append(0.0 if result is None else result[1] - result[0])
    residual = np.flatnonzero(np.asarray(widths) > 1e-8)
    if len(residual) == 0:
        return residual, np.zeros((0, 0))
    reduced = A[:, residual]
    _, singular, vh = np.linalg.svd(reduced, full_matrices=True)
    rank = int((singular > 1e-10).sum())
    return residual, vh[rank:].T


def combinatorial_cycle_basis(na: int, nb: int,
                              edges: list[tuple[int, int]],
                              residual: np.ndarray) -> np.ndarray:
    """Build an integer fundamental-cycle basis without SVD or optimization."""
    if len(residual) == 0:
        return np.zeros((0, 0))
    parent = list(range(na + nb))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        left = find(left); right = find(right)
        if left == right:
            return False
        parent[right] = left
        return True

    tree = [[] for _ in range(na + nb)]
    chords: list[tuple[int, int, int]] = []
    for position, edge_index in enumerate(residual):
        i, j = edges[int(edge_index)]
        left, right = i, na + j
        if union(left, right):
            tree[left].append((right, position))
            tree[right].append((left, position))
        else:
            chords.append((left, right, position))

    basis = np.zeros((len(residual), len(chords)))
    for column, (left, right, chord_position) in enumerate(chords):
        # The chord is traversed A->B; close it along the unique tree path B->A.
        stack = [(right, -1, [])]
        path = None
        while stack:
            node, previous, prefix = stack.pop()
            if node == left:
                path = prefix
                break
            for neighbor, edge_position in tree[node]:
                if neighbor != previous:
                    stack.append((neighbor, node,
                                  prefix + [(node, neighbor, edge_position)]))
        if path is None:
            raise RuntimeError("fundamental-cycle path not found")
        basis[chord_position, column] = 1.0
        for start, end, edge_position in path:
            basis[edge_position, column] = 1.0 if start < na else -1.0
    return basis


def zero_signature_subset_count(signature_rows: np.ndarray) -> int:
    """Count binary row subsets with zero total signature exactly.

    Meet in the middle avoids enumerating all 2^r selectors while retaining
    integer arithmetic.  Rows are the cycle signatures of residual cells.
    """
    rows = np.asarray(signature_rows, dtype=np.int64)
    dimension = rows.shape[1] if rows.ndim == 2 else 0

    def half_counts(half: np.ndarray) -> dict[tuple[int, ...], int]:
        counts: dict[tuple[int, ...], int] = {(0,) * dimension: 1}
        for row in half:
            additions: dict[tuple[int, ...], int] = {}
            values = tuple(int(value) for value in row)
            for signature, multiplicity in counts.items():
                shifted = tuple(left + right
                                for left, right in zip(signature, values))
                additions[shifted] = additions.get(shifted, 0) + multiplicity
            for signature, multiplicity in additions.items():
                counts[signature] = counts.get(signature, 0) + multiplicity
        return counts

    split = len(rows) // 2
    left = half_counts(rows[:split])
    right = half_counts(rows[split:])
    return sum(multiplicity * right.get(tuple(-value for value in signature), 0)
               for signature, multiplicity in left.items())


def binary_selector_census(total_edges: int, residual: np.ndarray,
                           cycle_basis: np.ndarray) -> dict[str, int | float]:
    """Count every nontrivial binary cell-selection query by exactness."""
    integer_basis = np.rint(cycle_basis).astype(np.int64)
    if np.max(np.abs(cycle_basis - integer_basis), initial=0.0) > 1e-10:
        raise RuntimeError("binary-selector census requires an integer cycle basis")
    residual_exact = zero_signature_subset_count(integer_basis)
    fixed_multiplier = 1 << (total_edges - len(residual))
    total_nontrivial = (1 << total_edges) - 2
    exact_nontrivial = residual_exact * fixed_multiplier - 2
    if cycle_basis.shape[1] and np.any(integer_basis.sum(axis=0) != 0):
        raise RuntimeError("query complements do not negate cycle signatures")
    return {
        "selectors": total_nontrivial,
        "exact": exact_nontrivial,
        "ambiguous": total_nontrivial - exact_nontrivial,
        "ambiguous_pct": 100 * (total_nontrivial - exact_nontrivial)
        / total_nontrivial,
    }


def binary_census_self_audit(rng: np.random.Generator,
                             cases: int = 128) -> dict[str, int]:
    """Cross-check meet-in-the-middle counts by direct bit enumeration."""
    failures = 0
    selectors = 0
    for _ in range(cases):
        rows = int(rng.integers(1, 13))
        dimensions = int(rng.integers(0, min(5, rows + 1)))
        signatures = rng.integers(-2, 3, size=(rows, dimensions), dtype=np.int64)
        counted = zero_signature_subset_count(signatures)
        direct = 0
        for mask in range(1 << rows):
            total = np.zeros(dimensions, dtype=np.int64)
            for row in range(rows):
                if mask & (1 << row):
                    total += signatures[row]
            direct += int(np.all(total == 0))
        failures += int(counted != direct)
        selectors += 1 << rows
    return {"cases": cases, "selectors": selectors, "failures": failures}


def uniform_binary_query_panel(edge_count: int, size: int,
                               seed: int) -> np.ndarray:
    """Return a seed-committed uniform panel closed under complementation."""
    if size % 2:
        raise ValueError("complement-paired panel size must be even")
    rng = np.random.default_rng(seed)
    half = rng.integers(0, 2, size=(size // 2, edge_count), dtype=np.int8)
    return np.vstack([half, 1 - half])


def cycle_basis_crosscheck(A: np.ndarray, queries: np.ndarray,
                           residual: np.ndarray, numeric_basis: np.ndarray,
                           na: int, nb: int,
                           edges: list[tuple[int, int]]) -> dict[str, int]:
    """Compare numerical and graph-theoretic quotient constructions."""
    graph_basis = combinatorial_cycle_basis(na, nb, edges, residual)
    kernel_failures = 0
    if graph_basis.shape[1]:
        kernel_product = np.sum(
            A[:, residual, None] * graph_basis[None, :, :], axis=1
        )
        kernel_failures = int(not np.isfinite(kernel_product).all() or
                              np.max(np.abs(kernel_product)) > 1e-10)
    dimension_failures = int(graph_basis.shape[1] != numeric_basis.shape[1])
    numeric_signatures = (np.zeros((len(queries), 0)) if numeric_basis.size == 0
                          else np.sum(queries[:, residual, None] *
                                      numeric_basis[None, :, :], axis=1))
    graph_signatures = (np.zeros((len(queries), 0)) if graph_basis.size == 0
                        else np.sum(queries[:, residual, None] *
                                    graph_basis[None, :, :], axis=1))
    finite_failures = int(not np.isfinite(numeric_signatures).all() or
                          not np.isfinite(graph_signatures).all())
    numeric_exact = (np.max(np.abs(numeric_signatures), axis=1) <= 1e-8
                     if numeric_signatures.shape[1] else np.ones(len(queries), dtype=bool))
    graph_exact = (np.max(np.abs(graph_signatures), axis=1) <= 1e-8
                   if graph_signatures.shape[1] else np.ones(len(queries), dtype=bool))
    exactness_failures = int(np.sum(numeric_exact != graph_exact))
    numeric_rank = int(np.linalg.matrix_rank(numeric_signatures, tol=1e-8))
    graph_rank = int(np.linalg.matrix_rank(graph_signatures, tol=1e-8))
    rank_failures = int(numeric_rank != graph_rank)
    return {
        "checks": 2 + len(queries),
        "failures": (kernel_failures + dimension_failures +
                     finite_failures + exactness_failures + rank_failures),
        "query_checks": len(queries),
        "numeric_rank": numeric_rank,
        "graph_rank": graph_rank,
    }


def relative_interior_flow(A: np.ndarray, b: np.ndarray,
                           residual: np.ndarray) -> np.ndarray:
    """Find a feasible flow with uniform positive slack on every residual edge."""
    n_edges = A.shape[1]
    objective = np.zeros(n_edges + 1)
    objective[-1] = -1.0
    inequalities = np.zeros((len(residual), n_edges + 1))
    for row, edge in enumerate(residual):
        inequalities[row, int(edge)] = -1.0
        inequalities[row, -1] = 1.0
    keep = list(range(A.shape[0] - 1))
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(len(residual)),
        A_eq=np.c_[A[keep], np.zeros(len(keep))],
        b_eq=b[keep],
        bounds=[(0.0, None)] * (n_edges + 1),
        method="highs",
    )
    if not result.success or result.x[-1] <= 1e-10:
        raise RuntimeError("failed to construct a relative-interior flow")
    return result.x[:-1]


def workload_disclosure_basis(queries: np.ndarray, residual: np.ndarray,
                              cycle_basis: np.ndarray, A: np.ndarray,
                              b: np.ndarray) -> dict[str, object]:
    """Choose a minimum-cardinality workload basis in cycle-signature space."""
    if cycle_basis.size == 0:
        return {"rank": 0, "selected": [], "checks": len(queries),
                "span_failures": 0, "minimality_failures": 0,
                "witness_worlds": 0, "witness_failures": 0,
                "oracle_worlds": 0, "oracle_failures": 0, "failures": 0}
    # Explicit contraction avoids a sporadic Accelerate/NumPy warning on
    # tiny rectangular matrix products while computing the same signatures.
    signatures = np.sum(
        queries[:, residual, None] * cycle_basis[None, :, :], axis=1
    )
    target_rank = int(np.linalg.matrix_rank(signatures, tol=1e-8))
    selected: list[int] = []
    selected_matrix = np.zeros((0, signatures.shape[1]))
    current_rank = 0
    for index, row in enumerate(signatures):
        candidate = np.vstack([selected_matrix, row])
        candidate_rank = int(np.linalg.matrix_rank(candidate, tol=1e-8))
        if candidate_rank > current_rank:
            selected.append(index)
            selected_matrix = candidate
            current_rank = candidate_rank
        if current_rank == target_rank:
            break
    combined_rank = int(np.linalg.matrix_rank(
        np.vstack([selected_matrix, signatures]), tol=1e-8
    ))
    span_failures = sum(
        int(np.linalg.matrix_rank(np.vstack([selected_matrix, row]), tol=1e-8)
            > target_rank)
        for row in signatures
    )
    minimality_failures = 0
    witness_failures = 0
    oracle_failures = 0
    interior = relative_interior_flow(A, b, residual)
    for position in range(len(selected)):
        reduced = np.delete(selected_matrix, position, axis=0)
        reduced_rank = int(np.linalg.matrix_rank(reduced, tol=1e-8))
        exposed = any(
            np.linalg.matrix_rank(np.vstack([reduced, row]), tol=1e-8)
            > reduced_rank for row in signatures
        )
        minimality_failures += int(reduced_rank != target_rank - 1 or not exposed)
        _, singular, vh = np.linalg.svd(reduced, full_matrices=True)
        rank = int((singular > 1e-10).sum())
        null_basis = vh[rank:].T
        removed_signature = selected_matrix[position]
        theta = null_basis @ (null_basis.T @ removed_signature)
        direction = np.sum(cycle_basis * theta[None, :], axis=1)
        direction_scale = float(np.max(np.abs(direction)))
        if direction_scale <= 1e-10:
            witness_failures += 2
            continue
        direction /= direction_scale
        nonzero = np.abs(direction) > 1e-10
        step_limit = float(np.min(interior[residual][nonzero] /
                                  np.abs(direction[nonzero])))
        step = 0.25 * step_limit
        worlds = []
        for sign in (-1.0, 1.0):
            world = interior.copy()
            world[residual] += sign * step * direction
            worlds.append(world)
        retained = [selected[index] for index in range(len(selected))
                    if index != position]
        retained_left = (np.sum(queries[retained] * worlds[0][None, :], axis=1)
                         if retained else np.zeros(0))
        retained_right = (np.sum(queries[retained] * worlds[1][None, :], axis=1)
                          if retained else np.zeros(0))
        same_disclosures = (not retained or np.max(np.abs(
            retained_left - retained_right)) <= 1e-7)
        removed_query = queries[selected[position]]
        separated = abs(float(np.sum(removed_query * worlds[0]) -
                              np.sum(removed_query * worlds[1]))) > 1e-7
        for world in worlds:
            valid = (float(world.min()) >= -1e-8 and
                     np.max(np.abs(np.sum(A * world[None, :], axis=1) - b)) <= 1e-7 and
                     same_disclosures and separated)
            witness_failures += int(not valid)
        # Independently optimize the removed query subject to the original
        # margins and all retained disclosure values.  This oracle never uses
        # the cycle basis or the perturbation direction above.
        keep = list(range(A.shape[0] - 1))
        retained_matrix = queries[retained] if retained else np.zeros((0, A.shape[1]))
        retained_values = (np.sum(retained_matrix * interior[None, :], axis=1)
                           if retained else np.zeros(0))
        equality_matrix = np.vstack([A[keep], retained_matrix])
        equality_values = np.r_[b[keep], retained_values]
        lower_result = linprog(removed_query, A_eq=equality_matrix,
                               b_eq=equality_values,
                               bounds=[(0.0, None)] * A.shape[1], method="highs")
        upper_result = linprog(-removed_query, A_eq=equality_matrix,
                               b_eq=equality_values,
                               bounds=[(0.0, None)] * A.shape[1], method="highs")
        strict_oracle_gap = (lower_result.success and upper_result.success and
                             -upper_result.fun - lower_result.fun > 1e-7)
        for result in (lower_result, upper_result):
            valid = (result.success and strict_oracle_gap and
                     float(result.x.min()) >= -1e-8 and
                     np.max(np.abs(np.sum(A * result.x[None, :], axis=1) - b)) <= 1e-7 and
                     (not retained or np.max(np.abs(
                         np.sum(retained_matrix * result.x[None, :], axis=1) -
                         retained_values)) <= 1e-7))
            oracle_failures += int(not valid)
    failures = (int(len(selected) != target_rank or combined_rank != target_rank)
                + span_failures + minimality_failures + witness_failures +
                oracle_failures)
    return {"rank": target_rank, "selected": selected,
            "checks": len(signatures) + 5 * len(selected),
            "span_failures": span_failures,
            "minimality_failures": minimality_failures,
            "witness_worlds": 2 * len(selected),
            "witness_failures": witness_failures,
            "oracle_worlds": 2 * len(selected),
            "oracle_failures": oracle_failures,
            "failures": failures}


def enumerate_integer_worlds(na: int, nb: int, edges: list[tuple[int, int]],
                             row: np.ndarray, col: np.ndarray,
                             upper: np.ndarray | None = None,
                             lower: np.ndarray | None = None) -> list[np.ndarray]:
    """Enumerate a tiny transportation fiber without calling an optimizer."""
    worlds: list[np.ndarray] = []
    values = np.zeros(len(edges), dtype=int)
    row_left = row.astype(int).copy()
    col_left = col.astype(int).copy()

    def visit(k: int) -> None:
        if k == len(edges):
            if not row_left.any() and not col_left.any():
                worlds.append(values.copy())
            return
        i, j = edges[k]
        later = edges[k + 1:]
        edge_lower = 0 if lower is None else int(lower[k])
        edge_upper = (min(row_left[i], col_left[j]) if upper is None else
                      min(row_left[i], col_left[j], int(upper[k])))
        for value in range(edge_lower, edge_upper + 1):
            values[k] = value
            row_left[i] -= value
            col_left[j] -= value
            row_possible = all(
                row_left[a] == 0 or any(u == a for u, _ in later)
                for a in range(na)
            )
            col_possible = all(
                col_left[b] == 0 or any(v == b for _, v in later)
                for b in range(nb)
            )
            if row_possible and col_possible:
                visit(k + 1)
            row_left[i] += value
            col_left[j] += value
        values[k] = 0

    visit(0)
    return worlds


def exhaustive_network_oracle(rng: np.random.Generator, cases: int = 512,
                              infeasible_cases: int = 256) -> dict[str, int]:
    """Challenge bounded LP results with complete integer-world enumeration."""
    endpoint_failures = exactness_failures = integrality_failures = 0
    dual_failures = 0; max_dual_error = 0.0
    transfer_failures = transfer_checks = 0
    transfer_rng = np.random.default_rng(4242)
    world_count = completed = 0
    while completed < cases:
        na = int(rng.integers(2, 4)); nb = int(rng.integers(2, 4))
        all_edges = [(i, j) for i in range(na) for j in range(nb)]
        edges = [edge for edge in all_edges if rng.random() < .75]
        if (not all(any(i == a for i, _ in edges) for a in range(na)) or
                not all(any(j == b for _, j in edges) for b in range(nb)) or
                len(edges) > 7):
            continue
        truth = rng.integers(0, 4, len(edges))
        if not truth.any():
            continue
        row = np.zeros(na, dtype=int); col = np.zeros(nb, dtype=int)
        for value, (i, j) in zip(truth, edges):
            row[i] += int(value); col[j] += int(value)
        lower = np.asarray([rng.integers(0, int(value) + 1) for value in truth])
        upper = truth + rng.integers(0, 4, len(edges))
        worlds = enumerate_integer_worlds(na, nb, edges, row, col, upper, lower)
        if not worlds:
            continue
        world_count += len(worlds); completed += 1
        A = incidence(na, nb, edges); b = np.r_[row, col].astype(float)
        c = rng.integers(-2, 4, len(edges)).astype(float)
        lo_result, hi_result = solve_pair(A, b, c, upper, lower)
        if not lo_result.success or not hi_result.success:
            endpoint_failures += 1
            continue
        lo, hi = float(lo_result.fun), float(-hi_result.fun)
        answers = np.asarray([float(c @ world) for world in worlds])
        endpoint_failures += int(abs(lo - answers.min()) > 1e-8 or
                                 abs(hi - answers.max()) > 1e-8)
        integrality_failures += int(abs(lo - round(lo)) > 1e-8 or
                                    abs(hi - round(hi)) > 1e-8)

        # Add an identified margin combination to the query.  Complete
        # enumeration, not either LP, then checks the quotient/signature
        # invariant: every world shifts by the same value and both arg-extreme
        # sets are unchanged.
        potential = transfer_rng.integers(-3, 4, A.shape[0]).astype(float)
        shifted_c = c + A.T @ potential
        delta = float(b @ potential)
        shifted_answers = np.asarray([float(shifted_c @ world) for world in worlds])
        shifted_result = solve_bounds(A, b, shifted_c, upper, lower)
        transfer_checks += 4
        if shifted_result is None:
            transfer_failures += 4
        else:
            shifted_lo, shifted_hi = shifted_result[:2]
            transfer_failures += int(np.max(np.abs(
                (shifted_answers - answers) - delta)) > 1e-8)
            transfer_failures += int(abs(shifted_lo - (lo + delta)) > 1e-8)
            transfer_failures += int(abs(shifted_hi - (hi + delta)) > 1e-8)
            base_arg = (set(np.flatnonzero(np.abs(answers - lo) <= 1e-8)),
                        set(np.flatnonzero(np.abs(answers - hi) <= 1e-8)))
            shifted_arg = (
                set(np.flatnonzero(np.abs(shifted_answers - shifted_lo) <= 1e-8)),
                set(np.flatnonzero(np.abs(shifted_answers - shifted_hi) <= 1e-8)),
            )
            transfer_failures += int(base_arg != shifted_arg)
        for objective, certificate in ((c, lo_result), (-c, hi_result)):
            error = endpoint_certificate_error(
                A, b, objective, certificate,
                lower.astype(float), upper.astype(float))
            max_dual_error = max(max_dual_error, error)
            dual_failures += int(error > 1e-7)
        residual, basis = cycle_exactness_index(A, b, upper, lower)
        cycle_exact = (basis.size == 0 or
                       float(np.max(np.abs(c[residual] @ basis))) <= 1e-8)
        exactness_failures += int(cycle_exact != (answers.min() == answers.max()))
    infeasible_failures = 0
    for trial in range(infeasible_cases):
        # Row zero has a single incident arc. Lowering its capacity by one
        # makes the balanced margins impossible without relying on an LP.
        edges = [(0, 0), (1, 0), (1, 1)]
        truth = rng.integers(1, 4, len(edges))
        row = np.array([truth[0], truth[1] + truth[2]], dtype=int)
        col = np.array([truth[0] + truth[1], truth[2]], dtype=int)
        lower = np.zeros(len(edges), dtype=int)
        upper = truth + rng.integers(1, 4, len(edges))
        if trial % 2 == 0:
            upper[0] = truth[0] - 1
        else:
            lower[0] = truth[0] + 1
        worlds = enumerate_integer_worlds(2, 2, edges, row, col, upper, lower)
        A = incidence(2, 2, edges); b = np.r_[row, col].astype(float)
        objective = rng.integers(-2, 4, len(edges)).astype(float)
        infeasible_failures += int(bool(worlds) or
                                   solve_bounds(A, b, objective, upper, lower) is not None)

    pd.DataFrame([{
        "cases": completed, "enumerated_worlds": world_count,
        "endpoint_failures": endpoint_failures,
        "exactness_failures": exactness_failures,
        "integrality_failures": integrality_failures,
        "dual_failures": dual_failures,
        "max_dual_error": max_dual_error,
        "signature_transfer_checks": transfer_checks,
        "signature_transfer_failures": transfer_failures,
        "infeasible_cases": infeasible_cases,
        "infeasible_failures": infeasible_failures,
    }]).to_csv(OUT / "exhaustive_network_audit.csv", index=False)
    return {
        "cases": completed, "worlds": world_count,
        "endpoint_failures": endpoint_failures,
        "exactness_failures": exactness_failures,
        "integrality_failures": integrality_failures,
        "dual_failures": dual_failures,
        "max_dual_error": max_dual_error,
        "signature_transfer_checks": transfer_checks,
        "signature_transfer_failures": transfer_failures,
        "infeasible_cases": infeasible_cases,
        "infeasible_failures": infeasible_failures,
    }


def exact_fields(county: pd.DataFrame, huc: pd.DataFrame) -> list[str]:
    metadata = {"state", "year", "agency", "scode", "area", "us"}
    fields = []
    for col in county.columns:
        if col in metadata or col not in huc.columns:
            continue
        a = pd.to_numeric(county[col], errors="coerce")
        b = pd.to_numeric(huc[col], errors="coerce")
        if (
            a.notna().all()
            and b.notna().all()
            and (a >= 0).all()
            and (b >= 0).all()
            and a.sum() > 0
            and abs(float(a.sum() - b.sum())) < 1e-7
        ):
            fields.append(col)
    return fields


def query_workload(edges, geometries) -> tuple[list[str], np.ndarray]:
    centroids = np.array([[geometries[e].centroid.x, geometries[e].centroid.y] for e in edges])
    x = centroids[:, 0]
    y = centroids[:, 1]
    queries: list[np.ndarray] = []
    names: list[str] = []
    for k in range(len(edges)):
        q = np.zeros(len(edges)); q[k] = 1
        queries.append(q); names.append(f"cell-{k + 1}")
    for axis_name, values in (("west", x), ("south", y)):
        for quantile in (0.2, 0.35, 0.5, 0.65, 0.8):
            cutoff = np.quantile(values, quantile)
            queries.append((values <= cutoff).astype(float))
            names.append(f"{axis_name}-{int(100 * quantile)}")
    for xq, yq in ((0.35, 0.35), (0.35, 0.65), (0.65, 0.35), (0.65, 0.65)):
        q = ((x <= np.quantile(x, xq)) & (y <= np.quantile(y, yq))).astype(float)
        queries.append(q); names.append(f"rectangle-{xq:.2f}-{yq:.2f}")
    return names, np.vstack(queries)


def fixed_spatial_specifications(edges, geometries) -> list[tuple[str, str, float, float]]:
    """Freeze non-cell predicates so support ablations retain their meaning."""
    centroids = np.array([[geometries[e].centroid.x,
                           geometries[e].centroid.y] for e in edges])
    x, y = centroids[:, 0], centroids[:, 1]
    specifications: list[tuple[str, str, float, float]] = []
    for axis_name, values in (("west", x), ("south", y)):
        for quantile in (0.2, 0.35, 0.5, 0.65, 0.8):
            specifications.append((
                f"{axis_name}-{int(100 * quantile)}", axis_name,
                float(np.quantile(values, quantile)), math.nan,
            ))
    for xq, yq in ((0.35, 0.35), (0.35, 0.65),
                   (0.65, 0.35), (0.65, 0.65)):
        specifications.append((
            f"rectangle-{xq:.2f}-{yq:.2f}", "rectangle",
            float(np.quantile(x, xq)), float(np.quantile(y, yq)),
        ))
    return specifications


def fixed_spatial_workload(edges, geometries, specifications
                           ) -> tuple[list[str], np.ndarray]:
    centroids = np.array([[geometries[e].centroid.x,
                           geometries[e].centroid.y] for e in edges])
    x, y = centroids[:, 0], centroids[:, 1]
    names = []
    queries = []
    for name, kind, x_cut, y_cut in specifications:
        if kind == "west":
            query = x <= x_cut
        elif kind == "south":
            query = y <= x_cut
        else:
            query = (x <= x_cut) & (y <= y_cut)
        names.append(name)
        queries.append(query.astype(float))
    return names, np.vstack(queries)


def plot_geometry(ax, geom, **kwargs):
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    patches = [MplPolygon(np.asarray(g.exterior.coords), closed=True) for g in geoms if not g.is_empty]
    ax.add_collection(PatchCollection(patches, **kwargs))


def make_map(counties, hucs, edges, geoms, xlo, xhi, query, lo, hi):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.0), constrained_layout=True)
    vmax = max(max(xlo), max(xhi), 1e-8)
    norm = Normalize(vmin=0.0, vmax=vmax)
    for panel, allocation, title in ((axes[0], None, "released supports"),
                                      (axes[1], xlo, f"sharp minimum\nquery = {lo:.2f}"),
                                      (axes[2], xhi, f"sharp maximum\nquery = {hi:.2f}")):
        for geom in counties.values():
            plot_geometry(panel, geom, facecolor="#f7f7f7", edgecolor="#333333", linewidth=.45)
        if allocation is not None:
            for k, edge in enumerate(edges):
                if query[k] > 0:
                    color = plt.cm.viridis(norm(allocation[k]))
                    plot_geometry(panel, geoms[edge], facecolor=color, edgecolor="none", alpha=.9)
        for geom in hucs.values():
            boundary = geom.boundary
            parts = [boundary] if boundary.geom_type == "LineString" else list(boundary.geoms)
            for line in parts:
                xy = np.asarray(line.coords); panel.plot(xy[:, 0], xy[:, 1], color="#b2182b", lw=.35)
        panel.set_title(title, fontsize=8); panel.set_aspect("equal"); panel.axis("off")
    scale = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    scale.set_array([])
    bar = fig.colorbar(scale, ax=axes[1:], orientation="horizontal",
                       fraction=.055, pad=.025, aspect=26)
    bar.set_label("allocated mass in selected overlap cells", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    fig.savefig(PLOTS / "endpoint_worlds.pdf", bbox_inches="tight")
    plt.close(fig)


def scaling_experiment(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for n in (12, 24, 48, 96, 160):
        for rep in range(7):
            na = nb = n
            edge_set = {(i, i) for i in range(n)} | {(i, (i + 1) % n) for i in range(n)}
            while len(edge_set) < 5 * n:
                edge_set.add((int(rng.integers(n)), int(rng.integers(n))))
            edges = sorted(edge_set); A = incidence(na, nb, edges)
            truth = rng.gamma(1.5, 1.0, len(edges))
            # Accumulate marginals directly. This avoids a sporadic Accelerate/
            # NumPy warning observed for the tiny dense matrix product on macOS
            # while computing exactly the same incidence-vector product.
            b = np.zeros(na + nb, dtype=float)
            for amount, (i, j) in zip(truth, edges):
                b[i] += amount
                b[na + j] += amount
            c = rng.integers(0, 2, len(edges)).astype(float)
            start = time.perf_counter(); result = solve_bounds(A, b, c)
            elapsed = 1000 * (time.perf_counter() - start)
            if result is not None:
                rows.append({"vertices": 2 * n, "edges": len(edges), "ms": elapsed})
    return pd.DataFrame(rows)


def operator_diagram() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 1.55), constrained_layout=True)
    labels = ["validate\nmanifest", "feasible\nflow / cut", "residual\ncycles", "exact value\nor sharp range"]
    colors = ["#e0ecf4", "#bfd3e6", "#9ebcda", "#8c96c6"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = .02 + i * .245
        ax.add_patch(plt.Rectangle((x, .44), .20, .38, facecolor=color, edgecolor="#333333", lw=.7))
        ax.text(x + .10, .63, label, ha="center", va="center", fontsize=7)
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + .242, .63), xytext=(x + .202, .63),
                        arrowprops=dict(arrowstyle="->", lw=.8, color="#333333"))
    ax.text(.5, .22, "cut, cycle, potentials, or primal--dual endpoints", ha="center", fontsize=7)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    fig.savefig(PLOTS / "operator_flow.pdf", bbox_inches="tight")
    plt.close(fig)


def agriculture_validation() -> dict[str, int | float | str]:
    """Replicate the same identification test on native USDA farm counts."""
    counties = read_gzip_features(RAW_AG / "counties_ne_ny_2020.geojson.gz", "GEOID")
    hucs = read_gzip_features(RAW_AG / "huc6_01.geojson.gz", "huc6")
    county_values = pd.read_csv(RAW_AG / "county_farms_2022.csv", dtype={"fips": str})
    huc_values = pd.read_csv(RAW_AG / "huc6_farms_2022.csv", dtype={"huc6": str})
    county_values["fips"] = county_values.fips.str.zfill(5)
    huc_values["huc6"] = huc_values.huc6.str.zfill(6)
    county_codes = county_values.fips.tolist()
    huc_codes = huc_values.huc6.tolist()

    # Region 01 is not a union of whole counties. Closing the accounting graph
    # therefore requires one explicit outside-region remainder rather than
    # silently dropping the portions of boundary counties outside the HUCs.
    from shapely.ops import unary_union
    region = unary_union([hucs[h] for h in huc_codes])
    edges: list[tuple[int, int]] = []
    areas: list[float] = []
    edge_geometries: dict[tuple[int, int], object] = {}
    for i, county in enumerate(county_codes):
        geom = counties[county]
        for j, huc in enumerate(huc_codes):
            overlap = geom.intersection(hucs[huc])
            area = overlap.area / 1_000_000
            if area >= 0.5:
                edges.append((i, j)); areas.append(area)
                edge_geometries[(i, j)] = overlap
        outside_geom = geom.difference(region)
        outside = outside_geom.area / 1_000_000
        if outside >= 0.5:
            edge = (i, len(huc_codes))
            edges.append(edge); areas.append(outside)
            edge_geometries[edge] = outside_geom

    columns = huc_codes + ["outside"]
    row = county_values.farms.to_numpy(float)
    released_huc = huc_values.farms.to_numpy(float)
    outside_total = float(row.sum() - released_huc.sum())
    col = np.r_[released_huc, outside_total]
    A = incidence(len(county_codes), len(columns), edges)
    b = np.r_[row, col]
    feasible = solve_bounds(A, b, np.zeros(len(edges)))
    if feasible is None:
        raise RuntimeError("USDA county--HUC6 farm-count release is infeasible")
    flow_feasibility_failures = int(transport_min_cost_integer(
        row, col, edges, np.zeros(len(edges))) is None)
    pd.DataFrame([{
        "domain": "USDA farm counts", "lp_feasible": True,
        "integer_flow_feasible": not bool(flow_feasibility_failures),
    }]).to_csv(OUT / "agriculture_flow_feasibility_audit.csv", index=False)

    residual, cycle_basis = cycle_exactness_index(A, b)
    area_allocation = np.zeros(len(edges))
    for i in range(len(county_codes)):
        incident = [k for k, (u, _) in enumerate(edges) if u == i]
        denom = sum(areas[k] for k in incident)
        for k in incident:
            area_allocation[k] = row[i] * areas[k] / denom

    # Reuse the water-use study's multiscale spatial workload and add every
    # released margin as a negative control.  Each control must have zero width;
    # otherwise the query compiler has failed a known invariant.
    spatial_names, spatial_queries = query_workload(edges, edge_geometries)
    names = list(spatial_names)
    families = ["cell" if name.startswith("cell-") else
                "sweep" if name.startswith(("west-", "south-")) else
                "rectangle" for name in spatial_names]
    queries = [query.copy() for query in spatial_queries]
    for i, code in enumerate(county_codes):
        queries.append(np.asarray([float(u == i) for u, _ in edges]))
        names.append(f"county-margin-{code}")
        families.append("margin-control")
    for j, code in enumerate(columns):
        queries.append(np.asarray([float(v == j) for _, v in edges]))
        names.append(f"huc-margin-{code}")
        families.append("margin-control")

    disclosure = workload_disclosure_basis(
        spatial_queries, residual, cycle_basis, A, b
    )
    basis_crosscheck = cycle_basis_crosscheck(
        A, np.asarray(spatial_queries), residual, cycle_basis,
        len(county_codes), len(columns), edges
    )
    graph_basis = combinatorial_cycle_basis(
        len(county_codes), len(columns), edges, residual)
    uniform_queries = uniform_binary_query_panel(
        len(edges), size=16384, seed=271828)
    if graph_basis.shape[1]:
        integer_basis = np.rint(graph_basis).astype(np.int64)
        uniform_signatures = np.empty(
            (len(uniform_queries), graph_basis.shape[1]), dtype=np.int64)
        for start in range(0, len(uniform_queries), 256):
            stop = min(start + 256, len(uniform_queries))
            uniform_signatures[start:stop] = np.sum(
                uniform_queries[start:stop, residual, None].astype(np.int64) *
                integer_basis[None, :, :], axis=1)
    else:
        uniform_signatures = np.zeros(
            (len(uniform_queries), 0), dtype=np.int64)
    uniform_exact = (np.all(uniform_signatures == 0, axis=1)
                     if uniform_signatures.shape[1]
                     else np.ones(len(uniform_queries), dtype=bool))
    uniform_complement_failures = int(np.sum(
        uniform_exact[:len(uniform_queries) // 2] !=
        uniform_exact[len(uniform_queries) // 2:]))
    pd.DataFrame({
        "query_id": np.arange(len(uniform_queries)),
        "selected_cells": uniform_queries.sum(axis=1),
        "cycle_exact": uniform_exact,
    }).to_csv(OUT / "agriculture_uniform_query_panel.csv", index=False)

    # Optimize a fixed subsample and replay it with the independent integer
    # flow implementation.  The full panel needs only the exact cycle test.
    uniform_oracle_rows = []
    uniform_oracle_failures = 0
    uniform_oracle_indices = np.linspace(
        0, len(uniform_queries) - 1, 256, dtype=int)
    for query_index in uniform_oracle_indices:
        query = uniform_queries[query_index].astype(float)
        lo, hi, _, _ = solve_bounds(A, b, query)
        flow_lo = transport_min_cost_integer(row, col, edges, query)
        flow_neg_hi = transport_min_cost_integer(row, col, edges, -query)
        if flow_lo is None or flow_neg_hi is None:
            flow_hi = math.nan
            failures = 2
        else:
            flow_hi = float(-flow_neg_hi)
            failures = (int(abs(float(flow_lo) - lo) > 1e-7) +
                        int(abs(flow_hi - hi) > 1e-7))
        failures += int(((hi - lo) <= 1e-8) != bool(uniform_exact[query_index]))
        uniform_oracle_failures += failures
        uniform_oracle_rows.append({
            "query_id": int(query_index),
            "selected_cells": int(query.sum()),
            "lo": lo, "hi": hi, "relative_width": (hi - lo) / float(row.sum()),
            "cycle_exact": bool(uniform_exact[query_index]),
            "integer_flow_lo": flow_lo,
            "integer_flow_hi": flow_hi,
            "failures": failures,
        })
    uniform_oracle = pd.DataFrame(uniform_oracle_rows)
    uniform_oracle.to_csv(
        OUT / "agriculture_uniform_query_oracle.csv", index=False)
    pd.DataFrame([{
        "domain": "USDA farm counts", **basis_crosscheck
    }]).to_csv(OUT / "agriculture_cycle_basis_crosscheck.csv", index=False)

    records = []
    complement_failures = 0
    complement_max_error = 0.0
    flow_endpoint_checks = 0
    flow_endpoint_failures = 0
    flow_endpoint_max_error = 0.0
    dual_certificate_checks = 0
    dual_certificate_failures = 0
    dual_certificate_max_error = 0.0
    certificate_mutation_checks = 0
    certificate_mutation_acceptances = 0
    for name, family, q in zip(names, families, queries):
        lo_result, hi_result = solve_pair(A, b, q)
        if not lo_result.success or not hi_result.success:
            raise RuntimeError("feasible USDA release failed endpoint optimization")
        lo, hi = float(lo_result.fun), float(-hi_result.fun)
        xlo, xhi = lo_result.x, hi_result.x
        lo_dual_error = endpoint_certificate_error(A, b, q, lo_result)
        hi_dual_error = endpoint_certificate_error(A, b, -q, hi_result)
        dual_certificate_checks += 2
        dual_certificate_failures += int(lo_dual_error > 1e-7)
        dual_certificate_failures += int(hi_dual_error > 1e-7)
        dual_certificate_max_error = max(
            dual_certificate_max_error, lo_dual_error, hi_dual_error)
        for objective, certificate in ((q, lo_result), (-q, hi_result)):
            mutation_errors = certificate_mutation_errors(
                A, b, objective, certificate)
            certificate_mutation_checks += len(mutation_errors)
            certificate_mutation_acceptances += sum(
                error <= 1e-7 for error in mutation_errors.values())
        flow_lo = transport_min_cost_integer(row, col, edges, q)
        flow_neg_hi = transport_min_cost_integer(row, col, edges, -q)
        if flow_lo is None or flow_neg_hi is None:
            flow_endpoint_failures += 2
            flow_lo_value = flow_hi_value = flow_error = np.nan
        else:
            flow_lo_value = float(flow_lo)
            flow_hi_value = float(-flow_neg_hi)
            flow_error = max(abs(flow_lo_value - lo),
                             abs(flow_hi_value - hi))
            flow_endpoint_max_error = max(flow_endpoint_max_error, flow_error)
            flow_endpoint_failures += int(abs(flow_lo_value - lo) > 1e-7)
            flow_endpoint_failures += int(abs(flow_hi_value - hi) > 1e-7)
        flow_endpoint_checks += 2
        signature = (0.0 if cycle_basis.size == 0 else float(np.max(np.abs(
            np.sum(q[residual, None] * cycle_basis, axis=0)))))
        point = float(q @ area_allocation)
        if family != "margin-control":
            complement = solve_bounds(A, b, 1.0 - q)
            if complement is None:
                complement_failures += 1
            else:
                error = max(abs(complement[0] - (row.sum() - hi)),
                            abs(complement[1] - (row.sum() - lo)))
                complement_max_error = max(complement_max_error, error)
                complement_failures += int(error > 1e-7)
        edge_index = int(name.split("-")[1]) - 1 if family == "cell" else None
        i, j = edges[edge_index] if edge_index is not None else (-1, -1)
        records.append({
            "query": name, "family": family,
            "county": county_codes[i] if i >= 0 else "",
            "huc6": columns[j] if j >= 0 else "",
            "area_km2": areas[edge_index] if edge_index is not None else np.nan,
            "lo": lo, "hi": hi, "width": hi - lo,
            "integer_flow_lo": flow_lo_value,
            "integer_flow_hi": flow_hi_value,
            "integer_flow_max_error": flow_error,
            "dual_certificate_max_error": max(lo_dual_error, hi_dual_error),
            "relative_width": (hi - lo) / float(row.sum()),
            "cycle_exact": signature <= 1e-8,
            "area_point": point,
            "area_point_outside": point < lo - 1e-8 or point > hi + 1e-8,
            "max_balance_residual": max(
                float(np.max(np.abs(incidence_apply(
                    len(county_codes), len(columns), edges, xlo) - b))),
                float(np.max(np.abs(incidence_apply(
                    len(county_codes), len(columns), edges, xhi) - b)))),
            "integral_endpoints": (abs(lo - round(lo)) <= 1e-7 and
                                   abs(hi - round(hi)) <= 1e-7),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "agriculture_query_bounds.csv", index=False)
    huc_area = incidence_apply(
        len(county_codes), len(columns), edges, area_allocation)[len(county_codes):]
    max_area_residual = float(np.max(np.abs(huc_area - col)))
    audits = pd.DataFrame([
        {"audit": "cycle exactness versus optimized width", "trials": len(frame),
         "failures": int(((frame.width <= 1e-8) != frame.cycle_exact).sum())},
        {"audit": "endpoint balance replay", "trials": 2 * len(frame),
         "failures": int((frame.max_balance_residual > 1e-7).sum())},
        {"audit": "integral endpoint witness", "trials": 2 * len(frame),
         "failures": int((~frame.integral_endpoints).sum())},
        {"audit": "published-margin negative controls",
         "trials": int((frame.family == "margin-control").sum()),
         "failures": int((frame.loc[frame.family == "margin-control", "width"] > 1e-8).sum())},
        {"audit": "query-complement endpoint identity",
         "trials": 2 * int((frame.family != "margin-control").sum()),
         "failures": complement_failures},
        {"audit": "minimum workload-disclosure basis",
         "trials": disclosure["checks"],
         "failures": disclosure["failures"]},
        {"audit": "combinatorial versus numerical cycle basis",
         "trials": basis_crosscheck["checks"],
         "failures": basis_crosscheck["failures"]},
        {"audit": "independent integer-flow feasibility and endpoints",
         "trials": 1 + flow_endpoint_checks,
         "failures": flow_feasibility_failures + flow_endpoint_failures},
        {"audit": "native primal-dual endpoint certificates",
         "trials": dual_certificate_checks,
         "failures": dual_certificate_failures},
        {"audit": "corrupted native certificate rejection",
         "trials": certificate_mutation_checks,
         "failures": certificate_mutation_acceptances},
        {"audit": "uniform binary-query panel complement symmetry",
         "trials": len(uniform_queries),
         "failures": uniform_complement_failures},
        {"audit": "uniform binary-query panel endpoint replay",
         "trials": 3 * len(uniform_oracle),
         "failures": uniform_oracle_failures},
    ])
    audits.to_csv(OUT / "agriculture_oracle_audit.csv", index=False)
    derived = frame[frame.family != "margin-control"]
    ambiguous = derived[derived.width > 1e-8]
    controls = frame[frame.family == "margin-control"]
    return {
        "AgCounties": len(county_codes),
        "AgHucs": len(huc_codes),
        "AgEdges": len(edges),
        "AgCycleRank": len(edges) - int(np.linalg.matrix_rank(A)),
        "AgQueries": len(frame),
        "AgDerivedQueries": len(derived),
        "AgControlQueries": len(controls),
        "AgAmbiguousPct": f"{100 * len(ambiguous) / len(derived):.1f}",
        "AgMedianWidth": f"{100 * ambiguous.relative_width.median():.2f}",
        "AgMaxWidth": f"{100 * frame.relative_width.max():.1f}",
        "AgAreaOutside": int(frame.area_point_outside.sum()),
        "AgCellAreaOutside": int(frame.loc[frame.family == "cell",
                                                 "area_point_outside"].sum()),
        "AgAreaResidual": f"{max_area_residual:.1f}",
        "AgAuditFailures": int(audits.failures.sum()),
        "AgComplementMaxError": f"{complement_max_error:.1e}",
        "AgDisclosureRank": disclosure["rank"],
        "AgDisclosureChecks": disclosure["checks"],
        "AgDisclosureFailures": disclosure["failures"],
        "AgDisclosureWitnessWorlds": disclosure["witness_worlds"],
        "AgDisclosureOracleWorlds": disclosure["oracle_worlds"],
        "AgBasisChecks": basis_crosscheck["checks"],
        "AgBasisFailures": basis_crosscheck["failures"],
        "AgFlowFeasibilityChecks": 1,
        "AgFlowEndpointChecks": flow_endpoint_checks,
        "AgFlowFailures": flow_feasibility_failures + flow_endpoint_failures,
        "AgFlowMaxError": f"{flow_endpoint_max_error:.1e}",
        "AgDualChecks": dual_certificate_checks,
        "AgDualFailures": dual_certificate_failures,
        "AgDualMaxError": f"{dual_certificate_max_error:.1e}",
        "AgMutationChecks": certificate_mutation_checks,
        "AgMutationAcceptances": certificate_mutation_acceptances,
        "AgOutside": int(outside_total),
        "AgUniformQueries": len(uniform_queries),
        "AgUniformExact": int(uniform_exact.sum()),
        "AgUniformAmbiguousPct": f"{100 * (1 - uniform_exact.mean()):.1f}",
        "AgUniformOracleQueries": len(uniform_oracle),
        "AgUniformOracleFailures": uniform_oracle_failures,
        "AgUniformMedianWidth":
            f"{100 * uniform_oracle.relative_width.median():.2f}",
    }


def main() -> None:
    operator_diagram()
    agriculture = agriculture_validation()
    county = pd.read_csv(RAW / "ct85co.txt", sep="\t", dtype={"area": str})
    huc = pd.read_csv(RAW / "ct85w8.txt", sep="\t", dtype={"area": str})
    county["area"] = county["area"].astype(str).str.zfill(3)
    huc["area"] = huc["area"].astype(str).str.zfill(8)
    county_codes, huc_codes, edges, areas, geoms = build_support()
    county = county.set_index("area").loc[county_codes]
    huc = huc.set_index("area").loc[huc_codes]
    A = incidence(len(county_codes), len(huc_codes), edges)
    rank = int(np.linalg.matrix_rank(A))
    cycle_rank = len(edges) - rank
    fields = exact_fields(county.reset_index(), huc.reset_index())
    qnames, queries = query_workload(edges, geoms)

    records = []
    cuts = []
    disclosure_rows = []
    basis_rows = []
    census_rows = []
    witness = None
    feasible_fields = []
    field_inputs = {}
    flow_feasibility_checks = 0
    flow_feasibility_failures = 0
    flow_endpoint_checks = 0
    flow_endpoint_failures = 0
    flow_endpoint_max_error = 0.0
    dual_certificate_checks = 0
    dual_certificate_failures = 0
    dual_certificate_max_error = 0.0
    certificate_mutation_checks = 0
    certificate_mutation_acceptances = 0
    flow_feasibility_rows = []
    for field in fields:
        r = county[field].to_numpy(float); s = huc[field].to_numpy(float); b = np.r_[r, s]
        feasibility = solve_bounds(A, b, np.zeros(len(edges)))
        scaled_r = np.rint(100 * r).astype(np.int64)
        scaled_s = np.rint(100 * s).astype(np.int64)
        if (np.max(np.abs(100 * r - scaled_r)) > 1e-7 or
                np.max(np.abs(100 * s - scaled_s)) > 1e-7):
            raise RuntimeError("USGS published decimals do not scale to cents")
        flow_feasible = transport_min_cost_integer(
            scaled_r, scaled_s, edges, np.zeros(len(edges))) is not None
        flow_feasibility_checks += 1
        flow_feasibility_failures += int(flow_feasible != (feasibility is not None))
        flow_feasibility_rows.append({
            "field": field, "lp_feasible": feasibility is not None,
            "integer_flow_feasible": flow_feasible,
        })
        if feasibility is None:
            cut = violated_neighbor_cut(r, s, edges)
            cuts.append({"field": field,
                         "counties": ",".join(county_codes[i] for i in cut["rows"]),
                         "hucs": ",".join(huc_codes[j] for j in cut["neighbors"]),
                         "required": cut["required"], "available": cut["available"],
                         "violation": cut["violation"]})
            continue
        feasible_fields.append(field)
        field_inputs[field] = (b, float(r.sum()))
        residual, cycle_basis = cycle_exactness_index(A, b)
        disclosure = workload_disclosure_basis(queries, residual, cycle_basis, A, b)
        basis_crosscheck = cycle_basis_crosscheck(
            A, np.asarray(queries), residual, cycle_basis,
            len(county_codes), len(huc_codes), edges
        )
        graph_basis = combinatorial_cycle_basis(
            len(county_codes), len(huc_codes), edges, residual)
        census = binary_selector_census(len(edges), residual, graph_basis)
        census_rows.append({
            "field": field, "residual_edges": len(residual),
            "cycle_rank": graph_basis.shape[1], **census,
        })
        basis_rows.append({"field": field, **basis_crosscheck})
        disclosure_rows.append({
            "field": field,
            "workload_rank": disclosure["rank"],
            "selected_queries": ",".join(qnames[index]
                                           for index in disclosure["selected"]),
            "validation_checks": disclosure["checks"],
            "span_failures": disclosure["span_failures"],
            "minimality_failures": disclosure["minimality_failures"],
            "witness_worlds": disclosure["witness_worlds"],
            "witness_failures": disclosure["witness_failures"],
            "oracle_worlds": disclosure["oracle_worlds"],
            "oracle_failures": disclosure["oracle_failures"],
            "failures": disclosure["failures"],
        })
        total = float(r.sum())
        for qname, q in zip(qnames, queries):
            lo_result, hi_result = solve_pair(A, b, q)
            if not lo_result.success or not hi_result.success:
                continue
            lo, hi = float(lo_result.fun), float(-hi_result.fun)
            xlo, xhi = lo_result.x, hi_result.x
            lo_dual_error = endpoint_certificate_error(A, b, q, lo_result)
            hi_dual_error = endpoint_certificate_error(A, b, -q, hi_result)
            dual_certificate_checks += 2
            dual_certificate_failures += int(lo_dual_error > 1e-7)
            dual_certificate_failures += int(hi_dual_error > 1e-7)
            dual_certificate_max_error = max(
                dual_certificate_max_error, lo_dual_error, hi_dual_error)
            for objective, certificate in ((q, lo_result), (-q, hi_result)):
                mutation_errors = certificate_mutation_errors(
                    A, b, objective, certificate)
                certificate_mutation_checks += len(mutation_errors)
                certificate_mutation_acceptances += sum(
                    error <= 1e-7 for error in mutation_errors.values())
            flow_lo = transport_min_cost_integer(scaled_r, scaled_s, edges, q)
            flow_neg_hi = transport_min_cost_integer(scaled_r, scaled_s, edges, -q)
            if flow_lo is None or flow_neg_hi is None:
                flow_endpoint_failures += 2
                flow_lo_value = flow_hi_value = flow_error = np.nan
            else:
                flow_lo_value = float(flow_lo) / 100.0
                flow_hi_value = float(-flow_neg_hi) / 100.0
                flow_error = max(abs(flow_lo_value - lo),
                                 abs(flow_hi_value - hi))
                flow_endpoint_max_error = max(flow_endpoint_max_error, flow_error)
                flow_endpoint_failures += int(abs(flow_lo_value - lo) > 1e-7)
                flow_endpoint_failures += int(abs(flow_hi_value - hi) > 1e-7)
            flow_endpoint_checks += 2
            signature_norm = (0.0 if cycle_basis.size == 0 else
                              float(np.max(np.abs(q[residual] @ cycle_basis))))
            rec = {"field": field, "query": qname, "lo": lo, "hi": hi,
                   "integer_flow_lo": flow_lo_value,
                   "integer_flow_hi": flow_hi_value,
                   "integer_flow_max_error": flow_error,
                   "width": hi - lo, "relative_width": (hi - lo) / total,
                   "support": int(q.sum()), "total": total,
                   "cycle_exact": signature_norm <= 1e-8,
                   "signature_norm": signature_norm,
                   "dual_certificate_max_error": max(
                       lo_dual_error, hi_dual_error),
                   "max_balance_residual": max(float(np.max(np.abs(A @ xlo - b))),
                                                float(np.max(np.abs(A @ xhi - b)))),
                   "min_endpoint_cell": min(float(xlo.min()), float(xhi.min())),
                   "max_objective_residual": max(abs(float(q @ xlo) - lo),
                                                     abs(float(q @ xhi) - hi))}
            records.append(rec)
            if field == "to-frtot" and (witness is None or rec["relative_width"] > witness[0]):
                witness = (rec["relative_width"], qname, q, xlo, xhi, lo, hi)

    results = pd.DataFrame(records)
    results.to_csv(OUT / "query_bounds.csv", index=False)
    pd.DataFrame(flow_feasibility_rows).to_csv(
        OUT / "flow_feasibility_audit.csv", index=False)
    pd.DataFrame({"field": feasible_fields}).to_csv(OUT / "feasible_fields.csv", index=False)
    cut_frame = pd.DataFrame(cuts)
    cut_frame.to_csv(OUT / "infeasible_cuts.csv", index=False)
    disclosure_frame = pd.DataFrame(disclosure_rows)
    disclosure_frame.to_csv(OUT / "workload_disclosure_basis.csv", index=False)
    basis_frame = pd.DataFrame(basis_rows)
    basis_frame.to_csv(OUT / "cycle_basis_crosscheck.csv", index=False)
    census_frame = pd.DataFrame(census_rows)
    census_frame.to_csv(OUT / "binary_query_census.csv", index=False)

    support_rows = []
    for threshold in (0.0, 0.1, 0.5, 1.0, 2.0):
        _, _, threshold_edges, _, _ = build_support(threshold)
        threshold_A = incidence(len(county_codes), len(huc_codes), threshold_edges)
        feasible = 0
        for field in fields:
            b = np.r_[county[field].to_numpy(float), huc[field].to_numpy(float)]
            feasible += int(solve_bounds(
                threshold_A, b, np.zeros(len(threshold_edges))) is not None)
        support_rows.append({"minimum_area_km2": threshold,
                             "edges": len(threshold_edges),
                             "feasible_fields": feasible})
    support_frame = pd.DataFrame(support_rows)
    support_frame.to_csv(OUT / "support_sensitivity.csv", index=False)

    # Keep physical sweep/rectangle cutoffs fixed while changing only the
    # declared positive-area support.  This prevents the ablation from silently
    # changing query semantics when an edge is added or removed.
    spatial_specifications = fixed_spatial_specifications(edges, geoms)
    spatial_names, baseline_spatial_queries = fixed_spatial_workload(
        edges, geoms, spatial_specifications)
    baseline_spatial_bounds = {}
    for field, (b, total) in field_inputs.items():
        for name, query in zip(spatial_names, baseline_spatial_queries):
            lo, hi, _, _ = solve_bounds(A, b, query)
            baseline_spatial_bounds[(field, name)] = (lo, hi, total)
    support_query_rows = []
    for threshold, relation in ((0.0, "expanded"), (2.0, "restricted")):
        _, _, threshold_edges, _, threshold_geometries = build_support(threshold)
        threshold_A = incidence(len(county_codes), len(huc_codes), threshold_edges)
        _, threshold_queries = fixed_spatial_workload(
            threshold_edges, threshold_geometries, spatial_specifications)
        for field, (b, _) in field_inputs.items():
            if solve_bounds(threshold_A, b,
                            np.zeros(len(threshold_edges))) is None:
                continue
            for name, query in zip(spatial_names, threshold_queries):
                alt_lo, alt_hi, _, _ = solve_bounds(threshold_A, b, query)
                base_lo, base_hi, total = baseline_spatial_bounds[(field, name)]
                if relation == "expanded":
                    contained = (alt_lo <= base_lo + 1e-7 and
                                 alt_hi >= base_hi - 1e-7)
                else:
                    contained = (alt_lo >= base_lo - 1e-7 and
                                 alt_hi <= base_hi + 1e-7)
                base_exact = base_hi - base_lo <= 1e-8
                alt_exact = alt_hi - alt_lo <= 1e-8
                support_query_rows.append({
                    "minimum_area_km2": threshold, "relation": relation,
                    "field": field, "query": name,
                    "baseline_lo": base_lo, "baseline_hi": base_hi,
                    "alternative_lo": alt_lo, "alternative_hi": alt_hi,
                    "containment_pass": contained,
                    "baseline_exact": base_exact, "alternative_exact": alt_exact,
                    "exactness_agrees": base_exact == alt_exact,
                    "absolute_relative_width_change": abs(
                        (alt_hi - alt_lo) - (base_hi - base_lo)) / total,
                })
    support_query_frame = pd.DataFrame(support_query_rows)
    support_query_frame.to_csv(OUT / "support_query_sensitivity.csv", index=False)
    support_query_summary = support_query_frame.groupby(
        ["minimum_area_km2", "relation"], as_index=False
    ).agg(
        pairs=("query", "size"),
        containment_failures=("containment_pass", lambda x: int((~x).sum())),
        exactness_agreement_pct=("exactness_agrees", lambda x: 100 * x.mean()),
        exactness_changes=("exactness_agrees", lambda x: int((~x).sum())),
        maximum_relative_width_change=("absolute_relative_width_change", "max"),
    )
    support_query_summary.to_csv(
        OUT / "support_query_sensitivity_summary.csv", index=False)
    expanded_support = support_query_summary.loc[
        support_query_summary.relation == "expanded"].iloc[0]
    restricted_support = support_query_summary.loc[
        support_query_summary.relation == "restricted"].iloc[0]

    # Exhaust every distinct support induced by an area-materiality cutoff.
    # If E_t={e: area(e)>=t}, the support changes only immediately after one
    # of the finitely many distinct positive overlap areas.  Fixed predicates
    # keep the same geometric cutoffs throughout, so adjacent feasible states
    # must obey interval containment as support edges are removed.
    _, _, complete_edges, complete_areas, complete_geometries = build_support(0.0)
    breakpoint_cutoffs = [0.0] + [
        float(np.nextafter(area, np.inf))
        for area in sorted(np.unique(complete_areas))
    ]
    support_path_rows = []
    support_path_bound_rows = []
    support_path_transition_rows = []
    previous_bounds = {}
    for state, cutoff in enumerate(breakpoint_cutoffs):
        keep = complete_areas >= cutoff
        state_edges = [edge for edge, retained in zip(complete_edges, keep)
                       if retained]
        state_geometries = {
            edge: complete_geometries[edge] for edge in state_edges
        }
        state_A = incidence(len(county_codes), len(huc_codes), state_edges)
        if state_edges:
            _, state_queries = fixed_spatial_workload(
                state_edges, state_geometries, spatial_specifications)
        else:
            support_path_rows.append({
                "state": state, "minimum_area_km2": cutoff,
                "edges": 0, "feasible_fields": 0,
            })
            previous_bounds = {}
            continue
        current_bounds = {}
        feasible_count = 0
        for field in fields:
            r = county[field].to_numpy(float)
            s = huc[field].to_numpy(float)
            b = np.r_[r, s]
            if solve_bounds(state_A, b, np.zeros(len(state_edges))) is None:
                continue
            feasible_count += 1
            total = float(r.sum())
            for name, query in zip(spatial_names, state_queries):
                lo, hi, _, _ = solve_bounds(state_A, b, query)
                key = (field, name)
                current_bounds[key] = (lo, hi, total)
                support_path_bound_rows.append({
                    "state": state, "minimum_area_km2": cutoff,
                    "edges": len(state_edges), "field": field,
                    "query": name, "lo": lo, "hi": hi,
                    "relative_width": (hi - lo) / total,
                    "exact": hi - lo <= 1e-8,
                })
                if key in previous_bounds:
                    old_lo, old_hi, _ = previous_bounds[key]
                    contained = (lo >= old_lo - 1e-7 and
                                 hi <= old_hi + 1e-7)
                    old_exact = old_hi - old_lo <= 1e-8
                    exact_persists = (not old_exact or
                                      (hi - lo <= 1e-8 and
                                       abs(lo - old_lo) <= 1e-7 and
                                       abs(hi - old_hi) <= 1e-7))
                    support_path_transition_rows.append({
                        "from_state": state - 1, "to_state": state,
                        "field": field, "query": name,
                        "outer_lo": old_lo, "outer_hi": old_hi,
                        "inner_lo": lo, "inner_hi": hi,
                        "containment_pass": contained,
                        "outer_exact": old_exact,
                        "exact_persistence_pass": exact_persists,
                    })
        support_path_rows.append({
            "state": state, "minimum_area_km2": cutoff,
            "edges": len(state_edges), "feasible_fields": feasible_count,
        })
        previous_bounds = current_bounds
    support_path_frame = pd.DataFrame(support_path_rows)
    support_path_bounds = pd.DataFrame(support_path_bound_rows)
    support_path_transitions = pd.DataFrame(support_path_transition_rows)
    support_path_frame.to_csv(OUT / "support_breakpoint_states.csv", index=False)
    support_path_bounds.to_csv(OUT / "support_breakpoint_bounds.csv", index=False)
    support_path_transitions.to_csv(
        OUT / "support_breakpoint_transitions.csv", index=False)

    # Adversarially delete each declared baseline overlap in turn.  Unlike the
    # nested cutoff path above, these restrictions are not ordered by area and
    # can target a topologically pivotal edge.  Any field that remains feasible
    # must have bounds contained in the baseline contract; a baseline-exact
    # answer must persist with the same value.
    edge_deletion_rows = []
    edge_deletion_summary_rows = []
    edge_deletion_feasibility_rows = []
    for removed_index, removed_edge in enumerate(edges):
        retained_edges = [edge for index, edge in enumerate(edges)
                          if index != removed_index]
        retained_geometries = {
            edge: geoms[edge] for edge in retained_edges
        }
        retained_A = incidence(
            len(county_codes), len(huc_codes), retained_edges)
        _, retained_queries = fixed_spatial_workload(
            retained_edges, retained_geometries, spatial_specifications)
        feasible_count = 0
        for field, (b, total) in field_inputs.items():
            feasibility = solve_bounds(
                retained_A, b, np.zeros(len(retained_edges)))
            baseline_edge_lower = float(results.loc[
                (results.field == field) &
                (results["query"] == f"cell-{removed_index + 1}"),
                "lo"].iloc[0])
            predicted_feasible = baseline_edge_lower <= 1e-8
            observed_feasible = feasibility is not None
            edge_deletion_feasibility_rows.append({
                "removed_edge": str(removed_edge), "field": field,
                "baseline_edge_lower": baseline_edge_lower,
                "predicted_feasible": predicted_feasible,
                "observed_feasible": observed_feasible,
                "criterion_pass": predicted_feasible == observed_feasible,
            })
            if not observed_feasible:
                continue
            feasible_count += 1
            for name, query in zip(spatial_names, retained_queries):
                lo, hi, _, _ = solve_bounds(retained_A, b, query)
                base_lo, base_hi, _ = baseline_spatial_bounds[(field, name)]
                contained = (lo >= base_lo - 1e-7 and
                             hi <= base_hi + 1e-7)
                base_exact = base_hi - base_lo <= 1e-8
                exact_persists = (not base_exact or
                                  (hi - lo <= 1e-8 and
                                   abs(lo - base_lo) <= 1e-7 and
                                   abs(hi - base_hi) <= 1e-7))
                strict_contraction = (lo > base_lo + 1e-7 or
                                      hi < base_hi - 1e-7)
                edge_deletion_rows.append({
                    "removed_edge": str(removed_edge),
                    "field": field, "query": name,
                    "baseline_lo": base_lo, "baseline_hi": base_hi,
                    "restricted_lo": lo, "restricted_hi": hi,
                    "containment_pass": contained,
                    "baseline_exact": base_exact,
                    "exact_persistence_pass": exact_persists,
                    "strict_contraction": strict_contraction,
                    "relative_width_reduction":
                        ((base_hi - base_lo) - (hi - lo)) / total,
                })
        edge_deletion_summary_rows.append({
            "removed_edge": str(removed_edge),
            "retained_edges": len(retained_edges),
            "feasible_fields": feasible_count,
        })
    edge_deletion_frame = pd.DataFrame(edge_deletion_rows)
    edge_deletion_summary = pd.DataFrame(edge_deletion_summary_rows)
    edge_deletion_feasibility = pd.DataFrame(
        edge_deletion_feasibility_rows)
    edge_deletion_frame.to_csv(
        OUT / "leave_one_overlap_edge_out.csv", index=False)
    edge_deletion_summary.to_csv(
        OUT / "leave_one_overlap_edge_out_summary.csv", index=False)
    edge_deletion_feasibility.to_csv(
        OUT / "leave_one_overlap_edge_feasibility.csv", index=False)

    # Exhaust every two-edge support restriction.  The set criterion predicts
    # feasibility from one optimization on the original polytope; rebuilding
    # the reduced graph is an independent observed check.  We also count the
    # genuinely coupled cases in which each edge can vanish separately but the
    # pair cannot vanish together.
    individually_removable = {
        (row.field, row.removed_edge): bool(row.observed_feasible)
        for row in edge_deletion_feasibility.itertuples(index=False)
    }
    pair_deletion_rows = []
    for first_index in range(len(edges)):
        for second_index in range(first_index + 1, len(edges)):
            removed_indices = {first_index, second_index}
            removed_edges = (edges[first_index], edges[second_index])
            retained_edges = [edge for index, edge in enumerate(edges)
                              if index not in removed_indices]
            retained_A = incidence(
                len(county_codes), len(huc_codes), retained_edges)
            removal_query = np.zeros(len(edges))
            removal_query[first_index] = 1.0
            removal_query[second_index] = 1.0
            for field, (b, _) in field_inputs.items():
                joint_minimum = solve_bounds(A, b, removal_query)[0]
                predicted_feasible = joint_minimum <= 1e-8
                observed_feasible = solve_bounds(
                    retained_A, b, np.zeros(len(retained_edges))) is not None
                separate_feasible = (
                    individually_removable[(field, str(removed_edges[0]))] and
                    individually_removable[(field, str(removed_edges[1]))])
                cut = None if observed_feasible else violated_neighbor_cut(
                    b[:len(county_codes)], b[len(county_codes):],
                    retained_edges)
                pair_deletion_rows.append({
                    "first_edge": str(removed_edges[0]),
                    "second_edge": str(removed_edges[1]),
                    "field": field,
                    "baseline_joint_minimum": joint_minimum,
                    "predicted_feasible": predicted_feasible,
                    "observed_feasible": observed_feasible,
                    "criterion_pass": predicted_feasible == observed_feasible,
                    "jointly_essential_despite_separate_removability":
                        separate_feasible and not observed_feasible,
                    "cut_counties": ("" if cut is None else
                        ",".join(county_codes[i] for i in cut["rows"])),
                    "cut_hucs": ("" if cut is None else
                        ",".join(huc_codes[j] for j in cut["neighbors"])),
                    "cut_violation": (0.0 if cut is None else cut["violation"]),
                    "cut_certificate_pass": (observed_feasible or
                        cut["violation"] > 1e-7),
                })
    pair_deletion_frame = pd.DataFrame(pair_deletion_rows)
    pair_deletion_frame.to_csv(
        OUT / "leave_two_overlap_edges_feasibility.csv", index=False)

    # Close the higher-order loophole without enumerating 2^|E| restrictions.
    # For each measure, R is the full set of individually removable edges. If
    # deleting all of R remains feasible, monotonicity proves that every subset
    # of R remains feasible. Otherwise a witnessed failing pair proves that the
    # minimum interaction order is exactly two.
    individually_removable_indices = {
        field: [index for index, edge in enumerate(edges)
                if individually_removable[(field, str(edge))]]
        for field in field_inputs
    }
    pair_synergy_fields = set(pair_deletion_frame.loc[
        pair_deletion_frame[
            "jointly_essential_despite_separate_removability"],
        "field"])
    interaction_rows = []
    for field, (b, _) in field_inputs.items():
        removable_indices = set(individually_removable_indices[field])
        retained_edges = [edge for index, edge in enumerate(edges)
                          if index not in removable_indices]
        retained_A = incidence(
            len(county_codes), len(huc_codes), retained_edges)
        all_removed_feasible = solve_bounds(
            retained_A, b, np.zeros(len(retained_edges))) is not None
        pair_witnessed = field in pair_synergy_fields
        cut = None if all_removed_feasible else violated_neighbor_cut(
            b[:len(county_codes)], b[len(county_codes):], retained_edges)
        classification = ("all-subsets-safe" if all_removed_feasible else
                          "minimum-interaction-order-2" if pair_witnessed else
                          "higher-order-unresolved")
        interaction_rows.append({
            "field": field,
            "individually_removable_edges": len(removable_indices),
            "delete_all_individually_removable_feasible":
                all_removed_feasible,
            "pair_synergy_witnessed": pair_witnessed,
            "classification": classification,
            "classification_pass": (all_removed_feasible != pair_witnessed),
            "cut_counties": ("" if cut is None else
                ",".join(county_codes[i] for i in cut["rows"])),
            "cut_hucs": ("" if cut is None else
                ",".join(huc_codes[j] for j in cut["neighbors"])),
            "cut_violation": (0.0 if cut is None else cut["violation"]),
            "cut_certificate_pass": (all_removed_feasible or
                cut["violation"] > 1e-7),
        })
    interaction_frame = pd.DataFrame(interaction_rows)
    interaction_frame.to_csv(
        OUT / "all_order_removable_edge_interactions.csv", index=False)

    complement_failures = 0
    complement_error = 0.0
    complement_audit = results
    query_by_name = dict(zip(qnames, queries))
    for row in complement_audit.itertuples(index=False):
        b, total = field_inputs[row.field]
        complement = 1.0 - query_by_name[row.query]
        comp = solve_bounds(A, b, complement)
        if comp is None:
            complement_failures += 1
            continue
        error = max(abs(comp[0] - (total - row.hi)),
                    abs(comp[1] - (total - row.lo)))
        complement_error = max(complement_error, error)
        complement_failures += int(error > 1e-7)

    # This is a compact plot.  Generate it at its printed width instead
    # of shrinking a text-width canvas, which would also shrink every label.
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), constrained_layout=True)
    relative_width = 100 * results.relative_width
    axes[0].hist(relative_width, bins=np.linspace(0, 100, 21),
                 color="#2166ac", edgecolor="white")
    axes[0].set_yscale("log")
    exact_share = 100 * float((results.width <= 1e-8).mean())
    axes[0].text(.97, .94, f"exact: {exact_share:.1f}%",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=6.5)
    axes[0].set(xlabel="sharp width / statewide total (%)",
                ylabel="field--query pairs (log count)")
    support_summary = results.groupby("support").relative_width.agg(
        median="median",
        q25=lambda values: values.quantile(.25),
        q75=lambda values: values.quantile(.75),
    )
    axes[1].fill_between(
        support_summary.index,
        100 * support_summary.q25,
        100 * support_summary.q75,
        color="#f4a582", alpha=.30, label="interquartile range",
    )
    axes[1].plot(support_summary.index, 100 * support_summary["median"],
                 "o-", color="#b2182b", ms=3, label="median")
    axes[1].set(xlabel="overlap cells selected", ylabel="relative width (%)")
    axes[1].legend(frameon=False, fontsize=6.2, loc="upper left",
                   handlelength=1.2, borderaxespad=.25)
    for axis in axes:
        axis.tick_params(labelsize=6.5, pad=1.5)
        axis.xaxis.label.set_size(7)
        axis.yaxis.label.set_size(7)
    fig.savefig(PLOTS / "query_widths.pdf", bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(2027)
    exhaustive = exhaustive_network_oracle(np.random.default_rng(1729))
    census_audit = binary_census_self_audit(np.random.default_rng(314159))
    scaling = scaling_experiment(rng)
    scaling.to_csv(OUT / "scaling.csv", index=False)
    summary = scaling.groupby("edges").ms.agg(["median", "min", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(3.45, 2.7), constrained_layout=True)
    ax.plot(summary.edges, summary["median"], "o-", color="#1b7837", ms=3)
    ax.fill_between(summary.edges, summary["min"], summary["max"], color="#a6dba0", alpha=.55)
    ax.set(xlabel="support edges", ylabel="paired bound time (ms)")
    fig.savefig(PLOTS / "scaling.pdf", bbox_inches="tight")
    plt.close(fig)

    current_hucs = read_features(RAW / "ct_huc8.geojson", "huc8")
    current_hucs = {h: current_hucs["01100007" if h == "01090005" else h] for h in huc_codes}
    counties_geom = read_features(RAW / "ct_counties_2010.geojson", "COUNTY")
    if witness:
        _, qname, q, xlo, xhi, lo, hi = witness
        make_map(counties_geom, current_hucs, edges, geoms, xlo, xhi, q, lo, hi)
    else:
        qname, lo, hi = "none", math.nan, math.nan

    exact_count = int((results.width <= 1e-8).sum())
    ambiguous = results[results.width > 1e-8]
    total_queries = len(results)
    # These diagnostics test whether the headline effect depends on one
    # convenient environmental measure or one predicate family.
    field_robustness_rows = []
    for omitted in sorted(results.field.unique()):
        retained = results.loc[results.field != omitted]
        retained_ambiguous = retained.loc[retained.width > 1e-8]
        field_robustness_rows.append({
            "omitted_field": omitted,
            "pairs": len(retained),
            "ambiguous_pct": 100 * float((retained.width > 1e-8).mean()),
            "ambiguous_median_relative_width_pct":
                100 * float(retained_ambiguous.relative_width.median()),
        })
    field_robustness = pd.DataFrame(field_robustness_rows)
    field_robustness.to_csv(OUT / "leave_one_measure_out.csv", index=False)
    field_concentration = results.assign(
        ambiguous=(results.width > 1e-8).astype(int)
    ).groupby("field", as_index=False).agg(
        ambiguous_pairs=("ambiguous", "sum"),
        relative_width_mass=("relative_width", "sum"),
    )
    field_concentration["ambiguous_share_pct"] = (
        100 * field_concentration.ambiguous_pairs /
        field_concentration.ambiguous_pairs.sum()
    )
    field_concentration["width_share_pct"] = (
        100 * field_concentration.relative_width_mass /
        field_concentration.relative_width_mass.sum()
    )
    field_concentration.to_csv(OUT / "measure_concentration.csv", index=False)
    query_family = results["query"].str.extract(
        r"^(cell|west|south|rectangle)", expand=False)
    family_ablation_rows = []
    for omitted in sorted(query_family.unique()):
        retained = results.loc[query_family != omitted]
        retained_ambiguous = retained.loc[retained.width > 1e-8]
        family_ablation_rows.append({
            "omitted_family": omitted,
            "pairs": len(retained),
            "ambiguous_pct": 100 * float((retained.width > 1e-8).mean()),
            "ambiguous_median_relative_width_pct":
                100 * float(retained_ambiguous.relative_width.median()),
        })
    family_ablation = pd.DataFrame(family_ablation_rows)
    family_ablation.to_csv(OUT / "query_family_ablation.csv", index=False)
    cycle_failures = int(((results.width <= 1e-8) != results.cycle_exact).sum())
    primal_failures = int(((results.max_balance_residual > 1e-7)
                           | (results.min_endpoint_cell < -1e-8)
                           | (results.max_objective_residual > 1e-7)).sum())
    cut_failures = int((cut_frame.violation <= 1e-7).sum())
    audit_rows = [
        {"audit": "cycle exactness versus optimized width", "trials": len(results),
         "failures": cycle_failures},
        {"audit": "endpoint primal and objective replay", "trials": 2 * len(results),
         "failures": primal_failures},
        {"audit": "native primal-dual endpoint certificates",
         "trials": dual_certificate_checks,
         "failures": dual_certificate_failures},
        {"audit": "corrupted native certificate rejection",
         "trials": certificate_mutation_checks,
         "failures": certificate_mutation_acceptances},
        {"audit": "infeasible neighbor-cut witness", "trials": len(cut_frame),
         "failures": cut_failures},
        {"audit": "query-complement endpoint identity",
         "trials": 2 * len(complement_audit),
         "failures": complement_failures},
        {"audit": "enumerated integer-fiber endpoints", "trials": exhaustive["cases"],
         "failures": exhaustive["endpoint_failures"]},
        {"audit": "enumerated integer-fiber exactness", "trials": exhaustive["cases"],
         "failures": exhaustive["exactness_failures"]},
        {"audit": "integral endpoint identity", "trials": 2 * exhaustive["cases"],
         "failures": exhaustive["integrality_failures"]},
        {"audit": "primal-dual endpoint certificate", "trials": 2 * exhaustive["cases"],
         "failures": exhaustive["dual_failures"]},
        {"audit": "signature-equivalent query transfer",
         "trials": exhaustive["signature_transfer_checks"],
         "failures": exhaustive["signature_transfer_failures"]},
        {"audit": "capacity-induced infeasibility", "trials": exhaustive["infeasible_cases"],
         "failures": exhaustive["infeasible_failures"]},
        {"audit": "minimum native workload-disclosure basis",
         "trials": int(disclosure_frame.validation_checks.sum()),
         "failures": int(disclosure_frame.failures.sum())},
        {"audit": "combinatorial versus numerical cycle basis",
         "trials": int(basis_frame.checks.sum()),
         "failures": int(basis_frame.failures.sum())},
        {"audit": "independent integer-flow feasibility and endpoints",
         "trials": flow_feasibility_checks + flow_endpoint_checks,
         "failures": flow_feasibility_failures + flow_endpoint_failures},
        {"audit": "fixed-predicate support interval containment",
         "trials": len(support_query_frame),
         "failures": int((~support_query_frame.containment_pass).sum())},
        {"audit": "all-breakpoint support interval containment",
         "trials": len(support_path_transitions),
         "failures": int((~support_path_transitions.containment_pass).sum())},
        {"audit": "all-breakpoint exact-answer persistence",
         "trials": int(support_path_transitions.outer_exact.sum()),
         "failures": int((support_path_transitions.outer_exact &
                           ~support_path_transitions.exact_persistence_pass).sum())},
        {"audit": "leave-one-overlap-edge interval containment",
         "trials": len(edge_deletion_frame),
         "failures": int((~edge_deletion_frame.containment_pass).sum())},
        {"audit": "edge-deletion feasibility iff zero lower endpoint",
         "trials": len(edge_deletion_feasibility),
         "failures": int((~edge_deletion_feasibility.criterion_pass).sum())},
        {"audit": "two-edge deletion iff zero joint lower endpoint",
         "trials": len(pair_deletion_frame),
         "failures": int((~pair_deletion_frame.criterion_pass).sum())},
        {"audit": "two-edge deletion neighbor-cut witness",
         "trials": int((~pair_deletion_frame.observed_feasible).sum()),
         "failures": int((~pair_deletion_frame.observed_feasible &
                           ~pair_deletion_frame.cut_certificate_pass).sum())},
        {"audit": "all-orders removable-edge interaction classification",
         "trials": len(interaction_frame),
         "failures": int((~interaction_frame.classification_pass).sum())},
        {"audit": "all-orders removable-edge cut witness",
         "trials": int((~interaction_frame[
             "delete_all_individually_removable_feasible"]).sum()),
         "failures": int((~interaction_frame[
             "delete_all_individually_removable_feasible"] &
             ~interaction_frame.cut_certificate_pass).sum())},
        {"audit": "leave-one-overlap-edge exact-answer persistence",
         "trials": int(edge_deletion_frame.baseline_exact.sum()),
         "failures": int((edge_deletion_frame.baseline_exact &
                           ~edge_deletion_frame.exact_persistence_pass).sum())},
        {"audit": "binary-query census versus direct enumeration",
         "trials": census_audit["selectors"],
         "failures": census_audit["failures"]},
    ]
    pd.DataFrame(audit_rows).to_csv(OUT / "oracle_audit.csv", index=False)
    summary = {
        "counties": len(county_codes),
        "hydrologic_units": len(huc_codes),
        "support_edges": len(edges),
        "cycle_rank": cycle_rank,
        "feasible_measures": len(feasible_fields),
        "query_predicates": len(qnames),
        "query_instances": total_queries,
        "exact_instances": exact_count,
        "ambiguous_fraction": 1 - exact_count / total_queries,
        "median_relative_width": float(results.relative_width.median()),
        "maximum_relative_width": float(results.relative_width.max()),
        "verification_failures": int(pd.DataFrame(audit_rows).failures.sum()),
        "witness": {"query": qname, "lower": lo, "upper": hi},
        "agricultural_validation": agriculture,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
