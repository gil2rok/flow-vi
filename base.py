from typing import NamedTuple, Protocol, Tuple

from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from jax.typing import ArrayLike


class SampleFn(Protocol):
    def __call__(
        self, rng_key: PRNGKey, parameters: ArrayLikeTree, num_samples: int
    ) -> ArrayTree: ...


class LogDensityFn(Protocol):
    def __call__(self, parameters: ArrayLikeTree, samples: ArrayTree) -> float: ...


class Distribution(NamedTuple):
    """Probability distribution with parameterized sample and log density methods."""

    sample: SampleFn
    log_density: LogDensityFn


class ForwardFn(Protocol):
    def __call__(
        self, x: ArrayLike, parameters: ArrayLikeTree
    ) -> Tuple[Array, float]: ...


class InverseFn(Protocol):
    def __call__(
        self, y: ArrayLike, parameters: ArrayLikeTree
    ) -> Tuple[Array, float]: ...


class Bijector(NamedTuple):
    """Parameterized bijective transformation with differentiable forward and inverse."""

    fwd_and_log_det: ForwardFn
    inv_and_log_det: InverseFn
