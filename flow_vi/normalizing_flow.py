from typing import List, Tuple

import jax
import jax.numpy as jnp
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from jax import random
from jax import scipy as jscipy
from jax.typing import ArrayLike

from .base import Bijector, Distribution


def RealNVP(flip_mask: bool) -> Bijector:
    """A RealNVP neural network bijector with masked affine coupling layers."""

    def _mlp(inputs: ArrayLike, params: ArrayLikeTree) -> Array:
        x = inputs
        for w, b in params[:-1]:
            a = jnp.dot(x, w) + b
            x = jax.nn.leaky_relu(a, negative_slope=0.01)
        w, b = params[-1]
        return jnp.dot(x, w) + b

    def _create_mask(x: Array) -> Array:
        d = x.shape[-1]
        mask = jnp.concatenate([jnp.ones(d // 2), jnp.zeros(d - d // 2)])
        return 1 - mask if flip_mask else mask

    def fwd_and_log_det(x: ArrayLike, params: ArrayLikeTree) -> Tuple[Array, float]:
        mask = _create_mask(x)
        x_masked = x * mask
        x_pass = x * (1 - mask)

        s = _mlp(x_masked, params["scale"])
        t = _mlp(x_masked, params["translate"])

        exp_s = jnp.exp(s)
        y = x_masked + (1 - mask) * (x_pass * exp_s + t)
        log_det = jnp.sum((1 - mask) * s, axis=-1)

        return y, log_det

    def inv_and_log_det(y: ArrayLike, params: ArrayLikeTree) -> Tuple[Array, float]:
        mask = _create_mask(y)
        y_masked = y * mask
        y_pass = y * (1 - mask)

        s = _mlp(y_masked, params["scale"])
        t = _mlp(y_masked, params["translate"])

        exp_neg_s = jnp.exp(-s)
        x = y_masked + (1 - mask) * (y_pass - t) * exp_neg_s
        log_det = -jnp.sum((1 - mask) * s, axis=-1)

        return x, log_det

    return Bijector(fwd_and_log_det, inv_and_log_det)


def compose_bijectors(bijectors: Tuple[Bijector, ...]) -> Bijector:
    """Compose multiple bijectors into a single bijector."""

    def fwd_and_log_det(x: ArrayLike, params: ArrayLikeTree) -> Tuple[Array, float]:
        total_log_det = 0.0
        for bijector, bijector_params in zip(bijectors, params):
            x, log_det = bijector.fwd_and_log_det(x, bijector_params)
            total_log_det += log_det
        return x, total_log_det

    def inv_and_log_det(y: ArrayLike, params: ArrayLikeTree) -> Tuple[Array, float]:
        total_log_det = 0.0
        for bijector, bijector_params in zip(reversed(bijectors), reversed(params)):
            y, log_det = bijector.inv_and_log_det(y, bijector_params)
            total_log_det += log_det
        return y, total_log_det

    return Bijector(fwd_and_log_det, inv_and_log_det)


def normalizing_flow(
    base_distribution: Distribution, base_parameters: ArrayLikeTree, bijector: Bijector
) -> Distribution:
    """Create a normalizing flow distribution using a base distribution and bijector."""

    def sample(
        rng_key: PRNGKey, bijector_parameters: ArrayLikeTree, num_samples: int
    ) -> ArrayTree:
        z = base_distribution.sample(rng_key, base_parameters, num_samples)
        x, _ = bijector.fwd_and_log_det(z, bijector_parameters)
        return x

    def log_density(bijector_parameters: ArrayLikeTree, x: Array) -> float:
        z, log_det = bijector.inv_and_log_det(x, bijector_parameters)
        log_density = base_distribution.log_density(base_parameters, z)
        return log_density - log_det

    return Distribution(sample, log_density)


def _init_layer_parameters(
    key: PRNGKey,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    init_scale: float,
) -> ArrayTree:
    """Initialize parameters for a RealNVP layer with scale and translate networks."""

    scale_key, translate_key = random.split(key)

    def init_network(net_key: PRNGKey) -> ArrayTree:
        keys = random.split(net_key, len(hidden_dims) + 1)
        dims = [input_dim] + hidden_dims + [output_dim]

        return [
            (random.normal(k, (m, n)) * init_scale, random.normal(k, (n,)) * init_scale)
            for k, m, n in zip(keys, dims[:-1], dims[1:])
        ]

    return {"scale": init_network(scale_key), "translate": init_network(translate_key)}


def init_flow_parameters(
    key: PRNGKey,
    num_layers: int,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    init_scale: float,
) -> ArrayTree:
    """Initialize parameters for a RealNVP normalizing flow."""

    keys = random.split(key, num_layers)
    return [
        _init_layer_parameters(k, input_dim, hidden_dims, output_dim, init_scale)
        for k in keys
    ]


def create_flow(
    key: PRNGKey,
    base_distribution: Distribution,
    base_parameters: ArrayLikeTree,
    num_layers: int,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    init_scale: float,
) -> Tuple[Distribution, ArrayTree]:
    """Create a normalizing flow distribution with multiple RealNVP layers."""

    # Alternate between flipping the mask for each RealNVP layer
    bijectors = [RealNVP(flip_mask=(i % 2 == 1)) for i in range(num_layers)]
    flow_bijector = compose_bijectors(bijectors)
    flow = normalizing_flow(base_distribution, base_parameters, flow_bijector)
    flow_parameters = init_flow_parameters(
        key, num_layers, input_dim, hidden_dims, output_dim, init_scale
    )
    return flow, flow_parameters


def create_gaussian(key: PRNGKey, dim: int) -> Tuple[Distribution, ArrayTree]:
    # TODO: implement reparameterization trick 
    sample = lambda k, p, n: random.multivariate_normal(k, *p, (n,))
    log_density = lambda p, s: jscipy.stats.multivariate_normal.logpdf(s, *p)
    gaussian = Distribution(sample, log_density)
    # gaussian_parameters = (random.normal(key, (dim,)), jnp.eye(dim))
    gaussian_parameters = (jnp.zeros(dim), jnp.eye(dim))
    return gaussian, gaussian_parameters
