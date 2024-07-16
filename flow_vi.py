from typing import Callable, NamedTuple, Protocol, Tuple

import jax
from jax import lax, jit
import jax.numpy as jnp
from optax import GradientTransformation, OptState

from blackjax.base import VIAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "FVIState",
    "FVIInfo",
    "sample",
    "generate_meanfield_logdensity",
    "step",
    "as_top_level_api",
]


class FVIState(NamedTuple):
    parameters: ArrayTree
    opt_state: OptState


class FVIInfo(NamedTuple):
    elbo: float


def init(
    initial_parameters: ArrayLikeTree,
    optimizer: GradientTransformation,
    *optimizer_args,
    **optimizer_kwargs,
) -> FVIState:
    """Initialize the flow VI state."""
    opt_state = optimizer.init(
        initial_parameters, 
        *optimizer_args, 
        **optimizer_kwargs
    )
    return FVIState(initial_parameters, opt_state)


def step(
    rng_key: PRNGKey,
    state: FVIState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 1000,
    stl_estimator: bool = True,
) -> tuple[FVIState, FVIInfo]:
    """Approximate the target density using flow VI.

    Parameters
    ----------
    rng_key
        Key for JAX's pseudo-random number generator.
    state
        Current state of the normalizing flow approximation.
    logdensity_fn
        Function that represents the target log-density to approximate.
    optimizer
        Optax `GradientTransformation` to be used for optimization.
    num_samples
        The number of samples that are taken from the approximation
        at each step to compute the Kullback-Leibler divergence between
        the approximation and the target log-density.
    stl_estimator
        Whether to use stick-the-landing (STL) gradient estimator :cite:p:`roeder2017sticking` for gradient estimation.
        The STL estimator has lower gradient variance by removing the score function term
        from the gradient. It is suggested by :cite:p:`agrawal2020advances` to always keep it in order for better results.

    """

    parameters = state.parameters

    # TODO
    def kl_divergence_fn(parameters):
        pass

    elbo, elbo_grad = jax.value_and_grad(kl_divergence_fn)(parameters)
    updates, new_opt_state = optimizer.update(elbo_grad, state.opt_state, parameters)
    new_parameters = jax.tree.map(lambda p, u: p + u, parameters, updates)
    new_state = FVIState(new_parameters, new_opt_state)
    return new_state, FVIInfo(elbo)


# TODO
def sample(rng_key: PRNGKey, state: FVIState, num_samples: int = 1):
    """Sample from the mean-field approximation."""
    pass


def as_top_level_api(
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 1000,
):
    """High-level implementation of Flow Variational Inference.

    Parameters
    ----------
    logdensity_fn
        A function that represents the log-density function associated with
        the distribution we want to sample from.
    optimizer
        Optax optimizer to use to optimize the ELBO.
    num_samples
        Number of samples to take at each step to optimize the ELBO.

    Returns
    -------
    A ``VIAlgorithm``.

    """

    def init_fn(position: ArrayLikeTree):
        return init(position, optimizer)

    def step_fn(rng_key: PRNGKey, state: FVIState) -> tuple[FVIState, FVIInfo]:
        return step(rng_key, state, logdensity_fn, optimizer, num_samples)

    def sample_fn(rng_key: PRNGKey, state: FVIState, num_samples: int):
        return sample(rng_key, state, num_samples)

    return VIAlgorithm(init_fn, step_fn, sample_fn)


class Bijection(Protocol):
    # why is one ArrayLikeTree and the other ArrayTree? Also should the return type be Tuple[ArrayTree, float]?
    def forward(self, x: ArrayLikeTree, params: ArrayTree) -> Tuple[Array, Array]:
        ...
    
    # why is one ArrayLikeTree and the other ArrayTree? Also should the return type be Tuple[ArrayTree, float]?
    def inverse(self, y: ArrayLikeTree, params: ArrayTree) -> Tuple[Array, Array]:
        ...


class Distribution(Protocol):
    def sample(self, rng_key: PRNGKey, params: ArrayTree, num_samples: int) -> ArrayTree:
        ...

    # should this return a float? depends on samples...
    def log_density(self, params: ArrayTree, samples: ArrayTree) -> ArrayTree:
        ...


class RealNVP(Bijection):
    def __init__(self, dim: int, hidden_dims: Tuple[int, ...]):
        self.dim = dim
        self.mask = jnp.arange(self.dim) % 2
        self.hidden_dims = hidden_dims

    @staticmethod
    @jax.jit
    def mlp(params: ArrayTree, inputs: ArrayLikeTree) -> Array:
        x = inputs
        for w, b in params[:-1]:
            x = jax.nn.relu(jnp.dot(x, w) + b)
        w, b = params[-1]
        return jnp.dot(x, w) + b

    @jax.jit
    def forward(self, x: ArrayLikeTree, params: ArrayTree) -> Tuple[Array, Array]:
        x_masked = x * self.mask
        x_pass = x * (1 - self.mask)

        s = self.mlp(params['scale'], x_masked)
        t = self.mlp(params['translate'], x_masked)

        exp_s = jnp.exp(s)
        y = x_masked + (1 - self.mask) * (x_pass * exp_s + t)
        log_det_jacobian = jnp.sum((1 - self.mask) * s, axis=-1)

        return y, log_det_jacobian

    @jax.jit
    def inverse(self, y: ArrayLikeTree, params: ArrayTree) -> Tuple[Array, Array]:
        y_masked = y * self.mask
        y_pass = y * (1 - self.mask)

        s = self.mlp(params['scale'], y_masked)
        t = self.mlp(params['translate'], y_masked)

        exp_neg_s = jnp.exp(-s)
        x = y_masked + (1 - self.mask) * (y_pass - t) * exp_neg_s
        log_det_jacobian = -jnp.sum((1 - self.mask) * s, axis=-1)

        return x, log_det_jacobian


# are all these jits necessary here?
@jit
def compose_bijections(bijections: Tuple[Bijection, ...]) -> Bijection:
    """Compose multiple bijections into a single bijection."""
    
    # why does this output Tupe[ArrayLikeTree, ArrayTree] instead of Tuple[ArrayTree, float]?
    @jit
    def forward(x: ArrayLikeTree, params: Tuple[ArrayTree, ...]) -> Tuple[ArrayLikeTree, ArrayTree]:
        def body_fn(carry, bijection_and_params):
            x, total_ldj = carry
            bijection, bijection_params = bijection_and_params
            y, ldj = bijection.forward(x, bijection_params)
            return (y, total_ldj + ldj), None

        init_ldj = jnp.zeros(jax.tree_util.tree_leaves(x)[0].shape[0])
        (y, total_ldj), _ = lax.scan(body_fn, (x, init_ldj), (bijections, params))
        return y, total_ldj

    # why does this output Tupe[ArrayLikeTree, ArrayTree] instead of Tuple[ArrayTree, float]?
    @jit
    def inverse(y: ArrayLikeTree, params: Tuple[ArrayTree, ...]) -> Tuple[ArrayLikeTree, ArrayTree]:
        def body_fn(carry, bijection_and_params):
            y, total_ldj = carry
            bijection, bijection_params = bijection_and_params
            x, ldj = bijection.inverse(y, bijection_params)
            return (x, total_ldj + ldj), None

        init_ldj = jnp.zeros(jax.tree_util.tree_leaves(y)[0].shape[0])
        (x, total_ldj), _ = lax.scan(body_fn, (y, init_ldj), (reversed(bijections), reversed(params)))
        return x, total_ldj

    return Bijection(forward=forward, inverse=inverse)


# TODO: consider NOT using this class if need complex access to forward and 
# inverse methods when computing the stick-the-landing gradient estimator
class NormalizingFlow(Distribution):
    def __init__(self, base_distribution: Distribution, bijection: Bijection):
        self.base_distribution = base_distribution
        self.bijection = bijection

    @jit
    def sample(self, rng_key: PRNGKey, params: ArrayLikeTree, num_samples: int) -> ArrayTree:
        base_params, bijection_params = params
        base_samples = self.base_distribution.sample(rng_key, base_params, num_samples)
        samples, _ = self.bijection.forward(base_samples, bijection_params) 
        return samples

    @jit
    def log_density(self, params: ArrayLikeTree, samples: ArrayLikeTree) -> ArrayTree:
        base_params, bijection_params = params
        base_x, ldj = self.bijection.inverse(samples, bijection_params)
        base_log_prob = self.base_distribution.log_density(base_params, base_x)
        return base_log_prob - ldj
