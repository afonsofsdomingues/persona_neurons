"""Baselines that tell us what "nothing" looks like.

A number such as "AUROC = 0.9" means little alone. A baseline runs the same
computation on something that should carry no persona signal, and the gap
between the two is the result.

Two baselines live here: persona vectors recomputed after shuffling the
condition labels, and uniformly random unit directions.
"""

from __future__ import annotations

import numpy as np

from persona_neurons.persona import difference_of_means


def shuffle_labels_within_groups(
    is_pos: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Randomly reassign the condition labels (trait-eliciting or
    trait-suppressing) among the responses to each question.

    Shuffling inside each question keeps the number of responses labelled
    trait-eliciting per question the same as in the real data, so the shuffled vector differs
    from the real one only in that its labels are unrelated to the persona.
    """
    shuffled = np.array(is_pos, copy=True)
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        shuffled[members] = rng.permutation(shuffled[members])
    return shuffled


def shuffled_label_vectors(
    acts: np.ndarray, is_pos: np.ndarray, groups: np.ndarray, n_shuffles: int, seed: int
) -> np.ndarray:
    """Difference-of-means vectors computed from shuffled labels.

    ``acts`` has shape ``[responses, layers, hidden]``. Returns an array of
    shape ``[n_shuffles, layers, hidden]``: one fake "persona vector" per
    shuffle and layer.

    How to read the baseline. A shuffled vector points in a direction chosen
    by chance, and its sign is arbitrary, so its AUROC is as likely to fall
    below 0.5 as above it. Note also that it is *not* a uniformly random
    direction: it is a random combination of the activations we collected, so
    it leans toward directions along which those activations vary most. If the
    persona contrast is one of those directions, shuffled vectors will partly
    align with it (with random sign) and their AUROCs will spread widely
    around 0.5. A wide shuffled band is therefore expected when the contrast is
    strong, and random directions are the cleaner noise floor.
    """
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(n_shuffles):
        fake_labels = shuffle_labels_within_groups(is_pos, groups, rng)
        vectors.append(difference_of_means(acts, fake_labels))
    return np.stack(vectors)


def random_unit_directions(n_directions: int, hidden_size: int, seed: int) -> np.ndarray:
    """Unit vectors pointing in uniformly random directions.

    Drawing each coordinate from a standard normal distribution and scaling to
    length 1 gives a direction with no preference for any axis. Unlike a
    shuffled-label vector, a random direction knows nothing about the data, so
    it shows how well an arbitrary direction separates the two sets of
    responses. Returns ``[n_directions, hidden_size]``.
    """
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_directions, hidden_size))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)
