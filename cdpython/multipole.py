from __future__ import annotations

import numpy as np

try:
    from ._common import (
        PreparedDataset,
        build_result,
        common_arg_parser,
        enumerate_one_sided_candidates,
        finalize_results,
        l2norm,
        prepare_dataset,
        print_results,
    )
except ImportError:
    from _common import (
        PreparedDataset,
        build_result,
        common_arg_parser,
        enumerate_one_sided_candidates,
        finalize_results,
        l2norm,
        prepare_dataset,
        print_results,
    )


def prepare_multipole_dataset(dataset_path: str) -> PreparedDataset:
    return prepare_dataset(dataset_path, l2norm)


def score_multipole(
    dataset_path: str,
    members: list[int] | tuple[int, ...],
) -> float:
    prepared = prepare_multipole_dataset(dataset_path)
    return score_multipole_from_prepared(prepared, members)


def score_multipole_from_prepared(
    prepared: PreparedDataset,
    members: list[int] | tuple[int, ...],
) -> float:
    matrix = prepared.processed_data[np.asarray(members, dtype=int)]
    corr = np.clip(matrix @ matrix.T, -1.0, 1.0)
    eigenvalues = np.linalg.eigvalsh(corr)
    return float(1.0 - np.min(eigenvalues))


def search_multipole(
    dataset_path: str,
    max_p: int = 2,
    threshold: float | None = None,
    top_k: int | None = None,
) -> list:
    prepared = prepare_multipole_dataset(dataset_path)
    results = (
        build_result(
            prepared,
            lhs=members,
            rhs=(),
            similarity=score_multipole_from_prepared(prepared, members),
        )
        for members in enumerate_one_sided_candidates(prepared.n, max_p=max_p)
    )
    return finalize_results(results, threshold=threshold, top_k=top_k)


def main() -> None:
    parser = common_arg_parser(
        "Exhaustive Multipole search using CorrelationDetective preprocessing.",
        two_sided=False,
    )
    args = parser.parse_args()
    results = search_multipole(
        args.dataset_path,
        max_p=args.max_p,
        threshold=args.threshold,
        top_k=args.top_k,
    )
    print_results(results)


if __name__ == "__main__":
    main()
