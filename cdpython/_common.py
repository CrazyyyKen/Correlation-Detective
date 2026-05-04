from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np


CD_VARIANCE_EPS = 1e-3
TOTAL_CORRELATION_BINS = 10


@dataclass(frozen=True)
class SearchResult:
    lhs: tuple[int, ...]
    rhs: tuple[int, ...]
    headers1: tuple[str, ...]
    headers2: tuple[str, ...]
    similarity: float


@dataclass(frozen=True)
class PreparedDataset:
    path: Path
    headers: tuple[str, ...]
    raw_data: np.ndarray
    processed_data: np.ndarray

    @property
    def n(self) -> int:
        return int(self.processed_data.shape[0])

    @property
    def m(self) -> int:
        return int(self.processed_data.shape[1])


def _read_column_major_csv(dataset_path: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    dataset_path = Path(dataset_path)
    with dataset_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Dataset is empty: {dataset_path}") from exc

        columns = [[] for _ in header]
        for row_idx, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Row {row_idx} in {dataset_path} has {len(row)} fields; expected {len(header)}."
                )
            for col_idx, value in enumerate(row):
                columns[col_idx].append(float(value))

    kept_headers: list[str] = []
    kept_columns: list[np.ndarray] = []
    for name, column in zip(header, columns):
        series = np.asarray(column, dtype=float)
        if np.std(series) >= CD_VARIANCE_EPS and (np.max(series) - np.min(series)) > 0:
            kept_headers.append(name)
            kept_columns.append(series)

    if not kept_columns:
        raise ValueError(
            "All columns were removed by the CorrelationDetective low-variance filter "
            f"(std < {CD_VARIANCE_EPS} or zero range)."
        )

    return tuple(kept_headers), np.vstack(kept_columns)


def l2norm(vector: np.ndarray) -> np.ndarray:
    centered = vector - np.mean(vector)
    norm = np.linalg.norm(centered)
    if norm == 0:
        raise ValueError("Encountered a zero-norm vector after mean-centering.")
    return centered / norm


def discretize(vector: np.ndarray, bins: int = TOTAL_CORRELATION_BINS) -> np.ndarray:
    minimum = float(np.min(vector))
    maximum = float(np.max(vector))
    width = maximum - minimum
    if width <= 0:
        raise ValueError("Encountered a constant vector during discretization.")

    out = np.floor((vector - minimum) / width * bins).astype(int)
    out[out == bins] = bins - 1
    return out


def prepare_dataset(
    dataset_path: str | Path,
    preprocess_row: Callable[[np.ndarray], np.ndarray],
) -> PreparedDataset:
    headers, raw_data = _read_column_major_csv(dataset_path)
    processed = np.vstack([preprocess_row(row.copy()) for row in raw_data])
    return PreparedDataset(
        path=Path(dataset_path),
        headers=headers,
        raw_data=raw_data,
        processed_data=processed,
    )


def average_weights(size: int) -> np.ndarray:
    return np.full(size, 1.0 / size, dtype=float)


def aggregate_rows(data: np.ndarray, indexes: Sequence[int]) -> np.ndarray:
    weights = average_weights(len(indexes))
    return np.tensordot(weights, data[np.asarray(indexes, dtype=int)], axes=(0, 0))


def enumerate_two_sided_candidates(
    n: int,
    max_p_left: int,
    max_p_right: int,
    allow_side_overlap: bool = False,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    universe = tuple(range(n))
    for left_size in range(1, max_p_left + 1):
        for lhs in itertools.combinations(universe, left_size):
            if allow_side_overlap:
                rhs_source = universe
            else:
                lhs_set = set(lhs)
                rhs_source = tuple(idx for idx in universe if idx not in lhs_set)
            for right_size in range(1, max_p_right + 1):
                for rhs in itertools.combinations(rhs_source, right_size):
                    yield lhs, rhs


def enumerate_one_sided_candidates(n: int, max_p: int) -> Iterator[tuple[int, ...]]:
    universe = tuple(range(n))
    for size in range(2, max_p + 1):
        yield from itertools.combinations(universe, size)


def build_result(
    prepared: PreparedDataset,
    lhs: Sequence[int],
    rhs: Sequence[int],
    similarity: float,
) -> SearchResult:
    return SearchResult(
        lhs=tuple(int(i) for i in lhs),
        rhs=tuple(int(i) for i in rhs),
        headers1=tuple(prepared.headers[i] for i in lhs),
        headers2=tuple(prepared.headers[i] for i in rhs),
        similarity=float(similarity),
    )


def finalize_results(
    results: Iterable[SearchResult],
    threshold: float | None = None,
    top_k: int | None = None,
) -> list[SearchResult]:
    materialized = list(results)
    if threshold is not None:
        materialized = [result for result in materialized if result.similarity >= threshold]
    materialized.sort(key=lambda result: result.similarity, reverse=True)
    if top_k is not None:
        materialized = materialized[:top_k]
    return materialized


def common_arg_parser(description: str, two_sided: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dataset_path", help="CSV path with a header row and one variable per column.")
    if two_sided:
        parser.add_argument("--max-p-left", type=int, default=1, help="Maximum LHS cardinality.")
        parser.add_argument("--max-p-right", type=int, default=1, help="Maximum RHS cardinality.")
        parser.add_argument(
            "--allow-side-overlap",
            action="store_true",
            help="Allow the same column to appear on both sides.",
        )
    else:
        parser.add_argument(
            "--max-p",
            type=int,
            default=2,
            help="Maximum candidate size. This matches CD's one-sided semantics.",
        )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional post-scoring threshold. No pruning is applied during search.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional post-scoring top-k cut after exhaustive evaluation.",
    )
    return parser


def print_results(results: Sequence[SearchResult]) -> None:
    for result in results:
        lhs = "-".join(map(str, result.lhs))
        rhs = "-".join(map(str, result.rhs))
        h1 = "-".join(result.headers1)
        h2 = "-".join(result.headers2)
        print(f"{lhs},{rhs},{h1},{h2},{result.similarity:.12f}")
