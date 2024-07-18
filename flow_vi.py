from typing import Callable, List, NamedTuple, Protocol, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import optax
from blackjax.base import VIAlgorithm
from blackjax.types import Array, ArrayTree, PRNGKey
from jax import lax, random
from optax import GradientTransformation, OptState

__all__ = [
    "FVIState",
    "FVIInfo",
    "sample",
    "generate_meanfield_logdensity",
    "step",
    "as_top_level_api",
]


class FVIState(NamedTuple):
    base_params: Tuple[Array, Array]
    bijector_params: List[dict]
    opt_state: OptState


class FVIInfo(NamedTuple):
    elbo: float


class Bijector(Protocol):
    """A parameterized bijective transformation with differentiable forward and inverse."""

    def fwd_and_log_det(self, x: Array, params: dict) -> Tuple[Array, Array]: ...

    def inv_and_log_det(self, y: Array, params: dict) -> Tuple[Array, Array]: ...


class Distribution(Protocol):
    """A probability distribution with sample and log density methods."""

    def sample(
        self, rng_key: PRNGKey, params: ArrayTree, num_samples: int
    ) -> ArrayTree: ...

    def log_density(self, params: ArrayTree, samples: ArrayTree) -> ArrayTree: ...


class StandardNormal(Distribution):
    @staticmethod
    def sample(
        rng_key: PRNGKey, params: Tuple[Array, Array], num_samples: int
    ) -> Array:
        mean, log_std = params
        return mean + jnp.exp(log_std) * random.normal(
            rng_key, (num_samples,) + mean.shape
        )

    @staticmethod
    def log_density(params: Tuple[Array, Array], samples: Array) -> Array:
        mean, log_std = params
        return jscipy.stats.norm.logpdf(samples, mean, jnp.exp(log_std)).sum(axis=-1)


class RealNVP(Bijector):
    """A RealNVP neural network bijector with masked affine coupling layers."""

    def __init__(self, dim: int):
        self.mask = jnp.arange(dim) % 2

    @staticmethod
    def mlp(inputs: Array, params: List[Tuple[Array, Array]]) -> Array:
        x = inputs
        for w, b in params[:-1]:
            x = jax.nn.relu(jnp.dot(x, w) + b)
        w, b = params[-1]
        return jnp.dot(x, w) + b

    def fwd_and_log_det(self, x: Array, params: dict) -> Tuple[Array, Array]:
        x_masked = x * self.mask
        x_pass = x * (1 - self.mask)

        s = self.mlp(params["scale"], x_masked)
        t = self.mlp(params["translate"], x_masked)

        exp_s = jnp.exp(s)
        y = x_masked + (1 - self.mask) * (x_pass * exp_s + t)
        log_det = jnp.sum((1 - self.mask) * s, axis=-1)

        return y, log_det

    def inv_and_log_det(self, y: Array, params: dict) -> Tuple[Array, Array]:
        y_masked = y * self.mask
        y_pass = y * (1 - self.mask)

        s = self.mlp(params["scale"], y_masked)
        t = self.mlp(params["translate"], y_masked)

        exp_neg_s = jnp.exp(-s)
        x = y_masked + (1 - self.mask) * (y_pass - t) * exp_neg_s
        log_det = -jnp.sum((1 - self.mask) * s, axis=-1)

        return x, log_det


class ComposedBijector(Bijector):
    """Compose multiple bijectors into a single bijector."""

    def __init__(self, bijectors: Tuple[Bijector, ...]):
        self.bijectors = bijectors

    def fwd_and_log_det(
        self, x: Array, params: Tuple[dict, ...]
    ) -> Tuple[Array, Array]:
        total_log_det = jnp.zeros_like(x)
        for bijector, bijector_params in zip(self.bijectors, params):
            x, log_det = bijector.fwd_and_log_det(x, bijector_params)
            total_log_det += log_det
        return x, total_log_det

    def inv_and_log_det(
        self, y: Array, params: Tuple[dict, ...]
    ) -> Tuple[Array, Array]:
        total_log_det = jnp.zeros_like(y)
        for bijector, bijector_params in zip(
            reversed(self.bijectors), reversed(params)
        ):
            y, log_det = bijector.inv_and_log_det(y, bijector_params)
            total_log_det += log_det
        return y, total_log_det


class NormalizingFlow(Distribution):
    """A normalizing flow distribution using a base distribution and a bijector."""

    def __init__(self, base_distribution: Distribution, bijector: Bijector):
        self.base_distribution = base_distribution
        self.bijector = bijector

    def sample(self, rng_key: PRNGKey, params: FVIState, num_samples: int) -> Array:
        base_params, bijector_params = params
        base_samples = self.base_distribution.sample(rng_key, base_params, num_samples)
        samples, _ = self.bijector.fwd_and_log_det(base_samples, bijector_params)
        return samples

    def log_density(self, params: FVIState, samples: Array) -> Array:
        base_params, bijector_params = params
        base_x, log_det = self.bijector.inv_and_log_det(samples, bijector_params)
        base_log_prob = self.base_distribution.log_density(base_params, base_x)
        return base_log_prob - log_det


def init_layer_params(
    key: PRNGKey, input_dim: int, hidden_dims: List[int], output_dim: int
) -> ArrayTree:
    """Initialize parameters for a single layer of RealNVP, including both scale and translate networks."""
    scale_key, translate_key = random.split(key)

    def init_network(net_key: PRNGKey) -> ArrayTree:
        keys = random.split(net_key, len(hidden_dims) + 1)
        dims = [input_dim] + hidden_dims + [output_dim]

        return [
            (random.normal(k, (m, n)) / jnp.sqrt(m), jnp.zeros(n))
            for k, m, n in zip(keys, dims[:-1], dims[1:])
        ]

    return {"scale": init_network(scale_key), "translate": init_network(translate_key)}


def init(
    rng_key: PRNGKey,
    dim: int,
    hidden_dims: List[int],
    num_layers: int,
    optimizer: GradientTransformation,
) -> FVIState:
    """Initialize the flow VI state."""
    base_params = (jnp.zeros(dim), jnp.zeros(dim))  # (mean, log_std)

    keys = random.split(rng_key, num_layers)
    bijector_params = [init_layer_params(key, dim, hidden_dims, dim) for key in keys]

    opt_state = optimizer.init(bijector_params)
    return FVIState(base_params, bijector_params, opt_state)


def step(
    rng_key: PRNGKey,
    state: FVIState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 1000,
    stl_estimator: bool = True,
) -> Tuple[FVIState, FVIInfo]:
    """Approximate the target density using flow VI."""

    def kl_divergence_fn(bijector_params):
        samples = sample(rng_key, state, num_samples)

        bijectors_list = [RealNVP(_get_pytree_input_dim(p)) for p in bijector_params]
        flow = NormalizingFlow(StandardNormal(), ComposedBijector(bijectors_list))
        log_q = flow.log_density((state.base_params, bijector_params), samples)

        log_p = logdensity_fn(samples)
        log_p = lax.stop_gradient(log_p) if stl_estimator else log_p

        kl = jnp.mean(log_q - log_p)
        return kl

    elbo, elbo_grad = jax.value_and_grad(kl_divergence_fn)(state.bijector_params)
    updates, new_opt_state = optimizer.update(
        elbo_grad, state.opt_state, state.bijector_params
    )
    new_bijector_params = jax.tree_util.tree.map(
        lambda p, u: p + u, state.bijector_params, updates
    )
    new_state = FVIState(state.base_params, new_bijector_params, new_opt_state)
    return new_state, FVIInfo(-elbo)


def sample(rng_key: PRNGKey, state: FVIState, num_samples: int = 1) -> Array:
    """Sample from the normalizing flow approximation of the target distribution."""
    bijectors_list = [RealNVP(_get_pytree_input_dim(p)) for p in state.bijector_params]
    flow = NormalizingFlow(StandardNormal(), ComposedBijector(bijectors_list))
    return flow.sample(rng_key, (state.base_params, state.bijector_params), num_samples)


def as_top_level_api(
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    dim: int,
    hidden_dims: List[int],
    num_layers: int,
    num_samples: int = 1000,
):
    """High-level implementation of Flow Variational Inference."""

    def init_fn(rng_key: PRNGKey) -> FVIState:
        return init(rng_key, dim, hidden_dims, num_layers, optimizer)

    def step_fn(rng_key: PRNGKey, state: FVIState) -> Tuple[FVIState, FVIInfo]:
        return step(rng_key, state, logdensity_fn, optimizer, num_samples)

    def sample_fn(rng_key: PRNGKey, state: FVIState, num_samples: int) -> Array:
        return sample(rng_key, state, num_samples)

    return VIAlgorithm(init_fn, step_fn, sample_fn)


def _get_pytree_input_dim(params: ArrayTree) -> int:
    """Extract the input dimension from a PyTree of parameters.

    This function assumes the input dimension is represented by the first
    dimension of the first array (weight matrix) found in the tree.
    """

    def is_array_leaf(x):
        return isinstance(x, (jnp.ndarray, Array))

    def get_first_array(tree):
        arrays = jax.tree.leaves(tree, is_leaf=is_array_leaf)
        if not arrays:
            raise ValueError("No arrays found in the parameter tree.")
        return arrays[0]

    first_array = get_first_array(params)
    if first_array.ndim < 2:
        raise ValueError("Expected at least 2D array for weight matrix.")
    return first_array.shape[0]


if __name__ == "__main__":
    # Define target distribution
    def target_log_prob(x):
        return -0.5 * jnp.sum(x**2)

    # Params
    dim = 2
    hidden_dims = [32, 32]
    num_layers = 5
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)

    vi_algorithm = as_top_level_api(
        target_log_prob, optimizer, dim, hidden_dims, num_layers
    )

    # Initialize
    rng_key = random.PRNGKey(0)
    state = init(rng_key, dim, hidden_dims, num_layers, optimizer)

    # Training loop
    # @jax.jit
    def train_step(rng_key, state):
        return step(rng_key, state, target_log_prob, optimizer)

    for i in range(1000):
        rng_key, subkey = random.split(rng_key)
        state, info = train_step(subkey, state)
        if i % 100 == 0:
            print(f"Step {i}, ELBO: {info.elbo}")

    # Sampling
    samples = sample(random.PRNGKey(1), state, num_samples=10000)
