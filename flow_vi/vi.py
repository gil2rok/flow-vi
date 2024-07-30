from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from blackjax.base import VIAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from jax import lax
from optax import GradientTransformation, OptState

from .base import Distribution

__all__ = [
    "VIState",
    "VIInfo",
    "sample",
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
    num_samples: int = 1000,
    stl_estimator: bool = False,
) -> Tuple[VIState, VIInfo]:
    """Perform a single update step to the parameters of the approximate distribution."""

    def negative_elbo_fn(parameters: ArrayTree) -> float:
        """Compute the negative Evidence Lower BOund (-ELBO), proportional to the KL 
        divergence between the approximate distribution and the target distribution."""
        samples = approximator.sample(rng_key, parameters, num_samples)
        if stl_estimator:  # TODO: confirm correct implementation
            parameters = jax.tree.map(lax.stop_gradient, parameters)
        
        log_q = jax.vmap(approximator.log_density, in_axes=(None, 0))(
            parameters, samples
        )
        log_p = jax.vmap(logdensity_fn)(samples)
        return jnp.mean(log_q - log_p)

    neg_elbo, neg_elbo_grad = jax.value_and_grad(negative_elbo_fn)(state.parameters)
    updates, new_opt_state = optimizer.update(
        neg_elbo_grad, state.opt_state, state.parameters
    )
    new_parameters = optax.apply_updates(state.parameters, updates)
    return VIState(new_parameters, new_opt_state), VIInfo(-neg_elbo)


def sample(
    rng_key: PRNGKey,
    parameters: ArrayLikeTree,
    approximator: Distribution,
    num_samples: int = 1,
) -> Array:
    """Sample from the normalizing flow approximation of the target distribution."""
    return approximator.sample(rng_key, parameters, num_samples)  # TODO: vmap or nah?


def as_top_level_api(
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    approximator: Distribution,
    batch_size: int,
) -> VIAlgorithm:
    """Create a VIAlgorithm for generic variational inference."""

    def init_fn(init_parameters: ArrayLikeTree) -> VIState:
        return init(init_parameters, optimizer)

    def step_fn(rng_key: PRNGKey, state: VIState) -> Tuple[VIState, VIInfo]:
        return step(rng_key, state, logdensity_fn, approximator, optimizer, batch_size)

    def sample_fn(
        rng_key: PRNGKey, parameters: ArrayLikeTree, num_samples: int = 1
    ) -> Array:
        return sample(rng_key, parameters, approximator, num_samples)

    return VIAlgorithm(init_fn, step_fn, sample_fn)
