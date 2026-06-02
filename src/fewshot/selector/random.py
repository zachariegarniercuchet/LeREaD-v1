"""Random few-shot selector — simple baseline."""
import random


def random_select_examples(
    examples: list[dict],
    n: int,
    seed: int = 42,
) -> tuple[list[int], list[dict]]:
    """
    Randomly sample `n` examples.
    Returns (selected_indices, selection_log).
    """
    rng     = random.Random(seed)
    indices = rng.sample(range(len(examples)), min(n, len(examples)))
    log     = [{"step": i + 1, "example_index": idx} for i, idx in enumerate(indices)]
    return indices, log