from __future__ import annotations

import numpy as np

try:
    from ._common import (
        PreparedDataset,
        aggregate_rows,
        build_result,
        common_arg_parser,
        enumerate_two_sided_candidates,
        finalize_results,
        l2norm,
        prepare_dataset,
        print_results,
    )
except ImportError:
    from _common import (
        PreparedDataset,
        aggregate_rows,
        build_result,
        common_arg_parser,
        enumerate_two_sided_candidates,
        finalize_results,
        l2norm,
        prepare_dataset,
        print_results,
    )


def prepare_euclidean_dataset(dataset_path: str) -> PreparedDataset:
    return prepare_dataset(dataset_path, l2norm)


def score_euclidean_similarity(
    dataset_path: str,
    lhs: list[int] | tuple[int, ...],
    rhs: list[int] | tuple[int, ...],
) -> float:
    prepared = prepare_euclidean_dataset(dataset_path)
    return score_euclidean_from_prepared(prepared, lhs, rhs)


def score_euclidean_from_prepared(
    prepared: PreparedDataset,
    lhs: list[int] | tuple[int, ...],
    rhs: list[int] | tuple[int, ...],
) -> float:
    left = l2norm(aggregate_rows(prepared.processed_data, lhs))
    right = l2norm(aggregate_rows(prepared.processed_data, rhs))
    distance = float(np.linalg.norm(left - right))
    return 1.0 / (1.0 + distance)


def search_euclidean_similarity(
    dataset_path: str,
    max_p_left: int = 1,
    max_p_right: int = 1,
    threshold: float | None = None,
    top_k: int | None = None,
    allow_side_overlap: bool = False,
) -> list:
    prepared = prepare_euclidean_dataset(dataset_path)
    results = (
        build_result(
            prepared,
            lhs,
            rhs,
            score_euclidean_from_prepared(prepared, lhs, rhs),
        )
        for lhs, rhs in enumerate_two_sided_candidates(
            prepared.n,
            max_p_left=max_p_left,
            max_p_right=max_p_right,
            allow_side_overlap=allow_side_overlap,
        )
    )
    return finalize_results(results, threshold=threshold, top_k=top_k)


def main() -> None:
    parser = common_arg_parser(
        "Exhaustive Euclidean similarity search using CorrelationDetective preprocessing.",
        two_sided=True,
    )
    args = parser.parse_args()
    results = search_euclidean_similarity(
        args.dataset_path,
        max_p_left=args.max_p_left,
        max_p_right=args.max_p_right,
        threshold=args.threshold,
        top_k=args.top_k,
        allow_side_overlap=args.allow_side_overlap,
    )
    print_results(results)


if __name__ == "__main__":
    main()
