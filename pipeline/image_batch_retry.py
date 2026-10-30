"""Adaptive image inference for GPU memory pressure."""

from typing import Any, Callable, Dict, List, Tuple


def predict_with_backoff(
    files: List[str], predict: Callable[[List[str], int], List[Any]],
    batch_size: int, empty_result: Any, clear_cache: Callable[[], None],
) -> Tuple[List[Any], List[Dict[str, str]]]:
    """Halve a failed batch until each image has succeeded or failed alone."""
    results: List[Any] = [empty_result for _ in files]
    failures: List[Dict[str, str]] = []

    def visit(start: int, end: int, size: int) -> None:
        subset = files[start:end]
        try:
            batch = predict(subset, max(1, min(size, len(subset))))
            if len(batch) != len(subset):
                raise RuntimeError(f"batch count mismatch: {len(batch)} != {len(subset)}")
            results[start:end] = batch
        except (RuntimeError, KeyError, ValueError, IndexError) as exc:
            clear_cache()
            if len(subset) == 1:
                failures.append({"image": subset[0], "error": f"{type(exc).__name__}: {exc}"[:300]})
                return
            middle = start + len(subset) // 2
            visit(start, middle, max(1, size // 2))
            visit(middle, end, max(1, size // 2))

    for start in range(0, len(files), max(1, batch_size)):
        visit(start, min(len(files), start + batch_size), batch_size)
    return results, failures
