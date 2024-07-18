import jax
import jax.numpy as jnp
import optax
from jax import random

from flow_vi.normalizing_flow import create_flow, create_gaussian
from flow_vi.vi import as_top_level_api


def main():
    rng_key = random.PRNGKey(0)

    # Define target distribution
    def logdensity_fn(samples):
        return -jnp.sum(samples**2, axis=-1) / 2  # Standard normal

    # Create approximator
    rng_key, gaussian_key, flow_key = random.split(rng_key, 3)
    gaussian, gaussian_parameters = create_gaussian(gaussian_key)
    flow, flow_parameters = create_flow(
        key=flow_key,
        base_distribution=gaussian,
        base_parameters=gaussian_parameters,
        num_layers=3,
        input_dim=2,
        hidden_dims=[16, 16],
        output_dim=2,
    )

    # Create VI algorithm
    vi = as_top_level_api(
        logdensity_fn=logdensity_fn,
        optimizer=optax.adam(1e-4),
        approximator=flow,
    )

    # Initialize
    state = vi.init(flow_parameters)

    # Training loop
    @jax.jit
    def train_step(rng_key, state):
        state, info = vi.step(rng_key, state)
        return state, info.elbo

    for i in range(10000):
        rng_key, subkey = random.split(rng_key)
        state, info = train_step(subkey, state)
        if i % 100 == 0:
            print(f"Step {i}, ELBO: {info}")

    # Generate sample from the flow
    rng_key, sample_key = random.split(rng_key)
    samples = vi.sample(sample_key, state.parameters, num_samples=100)

    # Compare true and approximated log densities
    true_logdensity = logdensity_fn(samples)
    approx_logdensity = flow.log_density(state.parameters, samples)
    mse = jnp.mean((true_logdensity - approx_logdensity) ** 2)
    print(f"MSE btwn true and approximated log densities: {mse}")


if __name__ == "__main__":
    main()
