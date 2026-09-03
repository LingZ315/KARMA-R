"""Small, deterministic learner family for a future prospective Replication B.

The module contains no target-data loader and performs no hyperparameter
selection. Hyperparameters must be supplied by a freeze-time configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _as_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return result


def _labels(value: np.ndarray, rows: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (rows,) or not np.isin(result, (0.0, 1.0)).all():
        raise ValueError("labels must be a binary vector aligned with the designs")
    return result


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def linear_design(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    query = _as_matrix(query, name="query")
    candidate = _as_matrix(candidate, name="candidate")
    if query.shape[0] != candidate.shape[0]:
        raise ValueError("query and candidate matrices must have the same row count")
    return np.column_stack((np.ones(query.shape[0]), query, candidate))


def bilinear_design(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    query = _as_matrix(query, name="query")
    candidate = _as_matrix(candidate, name="candidate")
    if query.shape[0] != candidate.shape[0]:
        raise ValueError("query and candidate matrices must have the same row count")
    interaction = np.einsum("ni,nj->nij", query, candidate).reshape(query.shape[0], -1)
    return np.column_stack((np.ones(query.shape[0]), query, candidate, interaction))


def _fit_logistic(design: np.ndarray, labels: np.ndarray, *, l2: float, maximum_iterations: int) -> np.ndarray:
    design = _as_matrix(design, name="design")
    labels = _labels(labels, design.shape[0])
    if l2 <= 0.0 or maximum_iterations < 1:
        raise ValueError("l2 and maximum_iterations must be positive")
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    for _ in range(maximum_iterations):
        probability = _sigmoid(design @ coefficients)
        score = design.T @ (labels - probability) - penalty @ coefficients
        weight = np.maximum(probability * (1.0 - probability), 1e-8)
        information = design.T @ (weight[:, None] * design) + penalty
        step = np.linalg.solve(information, score)
        coefficients += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("logistic fit produced non-finite coefficients")
    return coefficients


@dataclass
class LinearLogisticRouter:
    """L1: additive capability-conditioned regularized logistic reference."""

    l2: float = 1.0
    maximum_iterations: int = 100
    coefficients_: np.ndarray | None = None

    def fit(self, query: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> "LinearLogisticRouter":
        design = linear_design(query, candidate)
        self.coefficients_ = _fit_logistic(
            design, labels, l2=self.l2, maximum_iterations=self.maximum_iterations
        )
        return self

    def predict_proba(self, query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("router is not fitted")
        return _sigmoid(linear_design(query, candidate) @ self.coefficients_)


@dataclass
class BilinearLogisticRouter:
    """L2: explicit full bilinear query-by-candidate interaction model."""

    l2: float = 10.0
    maximum_iterations: int = 100
    coefficients_: np.ndarray | None = None
    query_dimension_: int | None = None
    candidate_dimension_: int | None = None

    def fit(self, query: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> "BilinearLogisticRouter":
        query = _as_matrix(query, name="query")
        candidate = _as_matrix(candidate, name="candidate")
        design = bilinear_design(query, candidate)
        self.coefficients_ = _fit_logistic(
            design, labels, l2=self.l2, maximum_iterations=self.maximum_iterations
        )
        self.query_dimension_ = query.shape[1]
        self.candidate_dimension_ = candidate.shape[1]
        return self

    @property
    def interaction_matrix_(self) -> np.ndarray:
        if self.coefficients_ is None or self.query_dimension_ is None or self.candidate_dimension_ is None:
            raise RuntimeError("router is not fitted")
        start = 1 + self.query_dimension_ + self.candidate_dimension_
        return self.coefficients_[start:].reshape(self.query_dimension_, self.candidate_dimension_)

    def predict_proba(self, query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("router is not fitted")
        return _sigmoid(bilinear_design(query, candidate) @ self.coefficients_)


@dataclass
class ResidualMLPRouter:
    """L3: compact nonlinear router with a linear skip connection."""

    hidden_dimension: int = 16
    l2: float = 1e-2
    learning_rate: float = 2e-2
    epochs: int = 400
    seed: int = 2026082810
    parameters_: dict[str, np.ndarray] | None = None

    def fit(self, query: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> "ResidualMLPRouter":
        query = _as_matrix(query, name="query")
        candidate = _as_matrix(candidate, name="candidate")
        if query.shape[0] != candidate.shape[0]:
            raise ValueError("query and candidate matrices must have the same row count")
        x = np.column_stack((query, candidate))
        y = _labels(labels, x.shape[0])
        if self.hidden_dimension < 1 or self.l2 <= 0 or self.learning_rate <= 0 or self.epochs < 1:
            raise ValueError("MLP hyperparameters must be positive")
        rng = np.random.Generator(np.random.PCG64(self.seed))
        scale = 1.0 / np.sqrt(max(x.shape[1], 1))
        p = {
            "w1": rng.normal(0.0, scale, size=(x.shape[1], self.hidden_dimension)),
            "b1": np.zeros(self.hidden_dimension),
            "w2": rng.normal(0.0, scale, size=self.hidden_dimension),
            "skip": np.zeros(x.shape[1]),
            "bias": np.zeros(1),
        }
        first = {key: np.zeros_like(value) for key, value in p.items()}
        second = {key: np.zeros_like(value) for key, value in p.items()}
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for step in range(1, self.epochs + 1):
            hidden = np.tanh(x @ p["w1"] + p["b1"])
            probability = _sigmoid(x @ p["skip"] + hidden @ p["w2"] + p["bias"][0])
            error = (probability - y) / len(y)
            hidden_error = error[:, None] * p["w2"][None, :] * (1.0 - hidden**2)
            gradients = {
                "w1": x.T @ hidden_error + self.l2 * p["w1"],
                "b1": hidden_error.sum(axis=0),
                "w2": hidden.T @ error + self.l2 * p["w2"],
                "skip": x.T @ error + self.l2 * p["skip"],
                "bias": np.asarray([error.sum()]),
            }
            for key in p:
                first[key] = beta1 * first[key] + (1.0 - beta1) * gradients[key]
                second[key] = beta2 * second[key] + (1.0 - beta2) * gradients[key] ** 2
                first_hat = first[key] / (1.0 - beta1**step)
                second_hat = second[key] / (1.0 - beta2**step)
                p[key] -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        if any(not value.shape or not np.isfinite(value).all() for value in p.values()):
            raise FloatingPointError("MLP fit produced invalid parameters")
        self.parameters_ = p
        return self

    def predict_proba(self, query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("router is not fitted")
        x = np.column_stack((_as_matrix(query, name="query"), _as_matrix(candidate, name="candidate")))
        p = self.parameters_
        hidden = np.tanh(x @ p["w1"] + p["b1"])
        return _sigmoid(x @ p["skip"] + hidden @ p["w2"] + p["bias"][0])


def architecture_sign_agreement(effects: Mapping[str, float], *, tolerance: float = 1e-12) -> str:
    """Return positive, negative, null, mixed, or insufficient without hiding disagreement."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if any(not np.isfinite(float(value)) for value in effects.values()):
        raise ValueError("architecture effects must be finite")
    if len(effects) < 2:
        return "insufficient"
    signs = {0 if abs(value) <= tolerance else (1 if value > 0 else -1) for value in effects.values()}
    if signs == {1}:
        return "positive"
    if signs == {-1}:
        return "negative"
    if signs == {0}:
        return "null"
    return "mixed"
