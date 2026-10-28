"""Private device-only logistic primal/dual gap diagnostics.

The dual variables are made feasible by reducing the larger of the two class
residual masses. This changes neither the coefficients nor the optimization
metric and therefore cannot introduce a coefficient null-space component.

The returned quantity is a primal/dual gap in exact arithmetic. Float32
arithmetic, GEMVs and transcendental functions do not provide a formal error
bound: ``roundoff_margin`` is an explicit numerical reserve, not a proof.
Callers must independently check the original-data gradient and must not use
``valid`` as a convergence decision by itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class DualityGapResult:
    """Lazy scalar diagnostics, retained on the selected MLX device."""

    gap: mx.array
    relative_gap: mx.array
    objective: mx.array
    dual_gradient_norm: mx.array
    feasibility_error: mx.array
    roundoff_margin: mx.array
    valid: mx.array


def _scaled_bernoulli_kl(logit, scale):
    """KL(scale * sigmoid(u) || sigmoid(u)) without entropy subtraction.

    Only scales in [0, 1] occur. The alternate complement expression keeps
    saturated positive logits finite and preserves the unchanged-scale zero.
    """
    q = mx.sigmoid(logit)
    remaining = mx.sigmoid(-logit) + (1.0 - scale) * q
    log_scale = mx.log(mx.maximum(scale, 1.1754943508222875e-38))
    # A zero scale multiplies a logarithm whose exact product is zero.
    log_scale = mx.where(scale > 0, log_scale, mx.zeros_like(log_scale))
    log_complement_ratio = mx.logaddexp(mx.zeros_like(logit), logit + mx.log1p(-scale))
    return scale * q * log_scale + remaining * log_complement_ratio


def logistic_duality_gap(
    X: mx.array,
    y: mx.array,
    theta: mx.array,
    sample_weight: mx.array,
    denominator: float,
    l2: float,
    *,
    device,
    logits: mx.array | None = None,
) -> DualityGapResult:
    """Evaluate a logistic primal/dual gap on a finite float32 design.

    ``theta`` has ``d`` entries, or ``d + 1`` with an unpenalized intercept.
    All inputs and outputs remain MLX arrays; scalar shape/device arguments
    are metadata. The design and optional logits must represent the same
    objective (a centered design and centered intercept are supported).
    The caller is responsible for validating finite design entries before
    fitting; nonfinite computed quantities invalidate this diagnostic.

    With sign ``1 - 2*y`` and normalized weights ``a``, the ordinary signed
    residual is ``sign * a * sigmoid(sign * logits)``. Reducing one class's
    residual mass enforces the intercept dual equality in exact arithmetic.
    The gap is the sum of weighted Bernoulli KL divergences and
    ``||l2 * w + X.T @ feasible_residual||**2 / (2*l2)``. Consequently it
    detects slow correlated directions that a componentwise gradient check
    alone can miss. The explicit feasibility error exposes float32 balance
    error; it is checked relative to the total dual mass.
    """
    if not isinstance(X, mx.array) or X.ndim != 2 or min(X.shape) == 0 or X.dtype != mx.float32:
        raise ValueError("X must be a nonempty float32 MLX matrix")
    n, d = X.shape
    if (
        not isinstance(y, mx.array) or y.shape != (n,)
        or not isinstance(theta, mx.array) or theta.shape not in ((d,), (d + 1,))
        or theta.dtype != mx.float32
        or not isinstance(sample_weight, mx.array) or sample_weight.shape != (n,)
        or sample_weight.dtype != mx.float32
    ):
        raise ValueError("labels, float32 parameters and float32 weights must match X")
    if logits is not None and (
        not isinstance(logits, mx.array) or logits.shape != (n,) or logits.dtype != mx.float32
    ):
        raise ValueError("logits must be a matching float32 MLX vector")
    for name, value in (("denominator", denominator), ("l2", l2)):
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 1.1754943508222875e-38 <= value <= 3.4028234663852886e38
        ):
            raise ValueError(f"{name} must be a normal positive float32 value")
    if device not in (mx.cpu, mx.gpu):
        raise ValueError("device must select the MLX CPU or GPU")
    with mx.stream(device):
        intercept = theta.size == d + 1
        raw = X @ theta[:d] if logits is None else logits
        if logits is None and intercept:
            raw = raw + theta[d]
        active = sample_weight > 0
        raw = mx.where(active, raw, mx.zeros_like(raw))
        sign = 1.0 - 2.0 * y.astype(mx.float32)
        signed = sign * raw
        normalized = sample_weight / denominator
        probability = mx.sigmoid(signed)
        beta = normalized * probability
        mass0 = mx.sum(mx.where(y == 0, beta, mx.zeros_like(beta)))
        mass1 = mx.sum(mx.where(y == 1, beta, mx.zeros_like(beta)))
        if intercept:
            target = mx.minimum(mass0, mass1)
            scale0 = mx.where(mass0 > 0, target / mx.maximum(mass0, 1.1754943508222875e-38), 1.0)
            scale1 = mx.where(mass1 > 0, target / mx.maximum(mass1, 1.1754943508222875e-38), 1.0)
            scale = mx.where(y == 0, scale0, scale1)
        else:
            scale = mx.ones_like(beta)
        feasible = sign * (scale * beta)
        dual_gradient = X.T @ feasible + l2 * theta[:d]
        gradient_squared = mx.sum(dual_gradient * dual_gradient)
        kl = mx.sum(normalized * _scaled_bernoulli_kl(signed, scale))
        # Each exact KL is nonnegative. Small negative cancellation in its
        # float32 expansion may not reduce the nonnegative gradient term.
        gap = mx.maximum(kl, 0.0) + gradient_squared / (2.0 * l2)
        loss = mx.sum(normalized * mx.logaddexp(mx.zeros_like(signed), signed))
        penalty = 0.5 * l2 * mx.sum(theta[:d] * theta[:d])
        objective = loss + penalty
        feasibility = mx.abs(mx.sum(feasible)) if intercept else mx.array(0.0, mx.float32)
        dual_mass = mx.sum(mx.abs(feasible))
        eps = 1.1920928955078125e-7
        # In finite arithmetic, a residual imbalance also leaves b * sum(r)
        # in the primal/dual identity. Expose that leakage instead of allowing
        # a large intercept to hide it behind a relative mass-only check.
        bias_leakage = mx.abs(theta[d]) * feasibility if intercept else mx.array(0.0, mx.float32)
        margin = 32.0 * eps * (mx.abs(objective) + penalty + mx.abs(kl)) + bias_leakage
        valid = (
            mx.all((y == 0) | (y == 1))
            & mx.all(mx.isfinite(sample_weight) & (sample_weight >= 0))
            & mx.any(active)
            & mx.all(mx.isfinite(raw)) & mx.all(mx.isfinite(theta))
            & mx.all(mx.isfinite(dual_gradient))
            & mx.isfinite(gap) & mx.isfinite(objective) & (objective > 0)
            & mx.isfinite(margin)
            & (feasibility <= 32.0 * eps * mx.maximum(dual_mass, 1.1754943508222875e-38))
        )
        gap = mx.where(valid, gap, float("inf"))
        relative = (gap + margin) / mx.maximum(objective, 1.1754943508222875e-38)
        return DualityGapResult(
            gap, relative, objective, mx.sqrt(gradient_squared), feasibility, margin, valid
        )
