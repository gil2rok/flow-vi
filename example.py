import multiprocessing
from typing import Callable

from blackjax.types import PRNGKey, ArrayLikeTree
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import optax
from jax import random
import matplotlib.pyplot as plt
import seaborn as sns

from flow_vi.normalizing_flow import create_flow, create_gaussian
from flow_vi.vi import as_top_level_api
from flow_vi import utils

sns.set_theme(style="whitegrid")


def main():
    # Setup
    num_iterations = 10_000
    batch_size = 512 * (2**10)
    rng_key = random.key(0)

    # Target distribution
    def logdensity_fn(samples):
        s1, s2 = jnp.split(samples, 2)
        return jscipy.stats.norm.logpdf(s1, 0, 10) + jscipy.stats.norm.logpdf(
            s2, 0.03 * (s1**2 - 100), 1
        )

    def sample_fn(key, num_samples):
        x_key, y_key = random.split(key)
        x = random.normal(x_key, (num_samples,)) * 10
        y = random.normal(y_key, (num_samples,)) + 0.03 * (x**2 - 100)
        return jnp.stack([x, y], axis=1)

    logdensity_dim = 2

    # Approximator
    rng_key, gaussian_key, flow_key = random.split(rng_key, 3)
    gaussian, gaussian_parameters = create_gaussian(gaussian_key, 2)
    flow, flow_parameters = create_flow(
        key=flow_key,
        base_distribution=gaussian,
        base_parameters=gaussian_parameters,
        num_layers=10,
        input_dim=logdensity_dim,
        hidden_dims=[32, 32],
        output_dim=logdensity_dim,
        init_scale=1e-3,
    )
    approximator, approximator_parameters = flow, flow_parameters

    # Optimizer
    scheduler = optax.cosine_decay_schedule(1e-3, num_iterations, 0)
    optimizer = optax.adam(learning_rate=scheduler)

    # Wasserstein distance
    def compute_wasserstein_distance(rng_key, parameters, n_samples=batch_size):
        true_samples = sample_fn(random.key(0), n_samples)  # fixed key
        approx_samples = vi.sample(rng_key, parameters, n_samples)
        return utils.wasserstein_distance(true_samples, approx_samples)

    # VI algorithm
    vi = as_top_level_api(
        logdensity_fn=logdensity_fn,
        optimizer=optimizer,
        approximator=approximator,
        batch_size=batch_size,
    )

    # Initialize VI state
    state = vi.init(approximator_parameters)

    # Train VI approximator
    @jax.jit
    def train_step(rng_key, state):
        state, info = vi.step(rng_key, state)
        return state, info.elbo

    for i in range(num_iterations):
        rng_key, subkey = random.split(rng_key)
        state, info = train_step(subkey, state)
        if i % 20 == 0:
            rng_key, subkey = random.split(rng_key)
            wass = compute_wasserstein_distance(subkey, state.parameters)
            print(f"Step {i}:\tELBO: {info:.6f}\tWasserstein: {wass:.6f}")

    # Generate samples from the approximator
    rng_key, sample_key = random.split(rng_key)
    approx_samples = vi.sample(sample_key, state.parameters, num_samples=10000)

    # Plot true vs approx samples
    true_samples = sample_fn(sample_key, 10000)
    _, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    sns.histplot(x=true_samples[:, 0], y=true_samples[:, 1], color="blue", ax=axs[0])
    sns.histplot(x=approx_samples[:, 0], y=approx_samples[:, 1], color="red", ax=axs[1])
    axs[0].set_title("True samples")
    axs[1].set_title("Approximated samples")
    plt.savefig("hist.png")


if __name__ == "__main__":
    ##### Run ON CPU #####
    # utils.set_platform("cpu")
    # utils.set_host_device_count(multiprocessing.cpu_count())
    # print(f"Running on {jax.device_count()} CPU cores.")

    ##### RUN ON GPU #####
    utils.set_platform("gpu")
    print(f"Running on {jax.device_count()} GPU cores.")

    main()
