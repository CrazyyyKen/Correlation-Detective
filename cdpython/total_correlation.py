from __future__ import annotations

import numpy as np

try:
    from ._common import (
        PreparedDataset,
        TOTAL_CORRELATION_BINS,
        build_result,
        common_arg_parser,
        discretize,
        enumerate_one_sided_candidates,
        finalize_results,
        prepare_dataset,
        print_results,
    )
except ImportError:
    from _common import (
        PreparedDataset,
        TOTAL_CORRELATION_BINS,
        build_result,
        common_arg_parser,
        discretize,
        enumerate_one_sided_candidates,
        finalize_results,
        prepare_dataset,
        print_results,
    )


def prepare_total_correlation_dataset(dataset_path: str) -> PreparedDataset:
    return prepare_dataset(dataset_path, discretize)


def entropy(values: np.ndarray, bins: int = TOTAL_CORRELATION_BINS) -> float:
    counts = np.bincount(values.astype(int), minlength=bins)
    probabilities = counts[counts > 0] / values.size
    return float(-np.sum(probabilities * np.log(probabilities)))


def joint_entropy(matrix: np.ndarray) -> float:
    if matrix.ndim == 1:
        return entropy(matrix)
    _, counts = np.unique(matrix.T, axis=0, return_counts=True)
    probabilities = counts / matrix.shape[1]
    return float(-np.sum(probabilities * np.log(probabilities)))


def score_total_correlation(
    dataset_path: str,
    members: list[int] | tuple[int, ...],
) -> float:
    prepared = prepare_total_correlation_dataset(dataset_path)
    return score_total_correlation_from_prepared(prepared, members)


def score_total_correlation_from_prepared(
    prepared: PreparedDataset,
    members: list[int] | tuple[int, ...],
) -> float:
    subset = prepared.processed_data[np.asarray(members, dtype=int)]
    return float(sum(entropy(row) for row in subset) - joint_entropy(subset))


def search_total_correlation(
    dataset_path: str,
    max_p: int = 2,
    threshold: float | None = None,
    top_k: int | None = None,
) -> list:
    prepared = prepare_total_correlation_dataset(dataset_path)
    results = (
        build_result(
            prepared,
            lhs=members,
            rhs=(),
            similarity=score_total_correlation_from_prepared(prepared, members),
        )
        for members in enumerate_one_sided_candidates(prepared.n, max_p=max_p)
    )
    return finalize_results(results, threshold=threshold, top_k=top_k)


def main() -> None:
    parser = common_arg_parser(
        "Exhaustive Total Correlation search using CorrelationDetective preprocessing.",
        two_sided=False,
    )
    args = parser.parse_args()
    results = search_total_correlation(
        args.dataset_path,
        max_p=args.max_p,
        threshold=args.threshold,
        top_k=args.top_k,
    )
    print_results(results)


if __name__ == "__main__":
    main()
