from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from blackjax.base import VIAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from jax import lax
from optax import GradientTransformation, OptState

from base import Distribution

__all__ = [
    "VIState",
    "VIInfo",
    "sample",
    "generate_meanfield_logdensity",
    "step",
    "as_top_level_api",
]


class VIState(NamedTuple):
    parameters: ArrayTree
    opt_state: OptState


class VIInfo(NamedTuple):
    elbo: float


def init(parameters: ArrayLikeTree, optimizer: GradientTransformation) -> VIState:
    """Initialize the variational inference state."""

    opt_state = optimizer.init(parameters)
    return VIState(parameters, opt_state)


def step(
    rng_key: PRNGKey,
    state: VIState,
    logdensity_fn: Callable,
    approximator: Distribution,
    optimizer: GradientTransformation,
    num_samples: int = 10000,
    stl_estimator: bool = False,
) -> Tuple[VIState, VIInfo]:
    """Perform a single update step to the parameters of the approximate distribution."""

    def kl_divergence_fn(parameters: ArrayTree) -> float:
        samples = sample(rng_key, parameters, approximator, num_samples)
        parameters = (
            jax.tree.map(lax.stop_gradient, parameters) if stl_estimator else parameters
        )  # TODO: confirm correct implementation
        approx_logdensity_fn = lambda s: approximator.log_density(parameters, s)

        log_q = jax.vmap(approx_logdensity_fn)(samples)
        log_p = jax.vmap(logdensity_fn)(samples)
        return jnp.mean(log_q - log_p)

    elbo, elbo_grad = jax.value_and_grad(kl_divergence_fn)(state.parameters)
    updates, new_opt_state = optimizer.update(
        elbo_grad, state.opt_state, state.parameters
    )
    new_parameters = jax.tree.map(lambda p, u: p + u, state.parameters, updates)
    return VIState(new_parameters, new_opt_state), VIInfo(-elbo)


def sample(
    rng_key: PRNGKey,
    parameters: ArrayLikeTree,
    approximator: Distribution,
    num_samples: int = 1,
) -> Array:
    """Sample from the normalizing flow approximation of the target distribution."""
    return approximator.sample(rng_key, parameters, num_samples)


def as_top_level_api(
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    approximator: Distribution,
) -> VIAlgorithm:
    """Create a VIAlgorithm for generic variational inference."""

    def init_fn(init_parameters: ArrayLikeTree) -> VIState:
        return init(init_parameters, optimizer)

    def step_fn(rng_key: PRNGKey, state: VIState) -> Tuple[VIState, VIInfo]:
        return step(rng_key, state, logdensity_fn, approximator, optimizer)

    def sample_fn(
        rng_key: PRNGKey, parameters: ArrayLikeTree, num_samples: int = 1
    ) -> Array:
        return sample(rng_key, parameters, approximator, num_samples)

    return VIAlgorithm(init_fn, step_fn, sample_fn)
