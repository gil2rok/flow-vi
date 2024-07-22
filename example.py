import multiprocessing

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
    num_iterations = 1_000 
    batch_size = 512 * (2 ** 10)
    rng_key = random.PRNGKey(0)

    # Target distribution
    def logdensity_fn(samples):
        return jscipy.stats.multivariate_normal.logpdf(
            samples, mean=jnp.array([10.0, 10.0]), cov=jnp.eye(2) * 5.0
        )
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
        init_scale=0.001,
    )
    approximator, approximator_parameters = flow, flow_parameters
    
    # Optimizer
    scheduler = optax.cosine_decay_schedule(1e-4, num_iterations, 0.0)
    optimizer = optax.adam(learning_rate=scheduler)

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
        if i % 100 == 0:
            print(f"Step {i}, ELBO: {info}")

    # Generate samples from the approximator
    rng_key, sample_key = random.split(rng_key)
    approx_samples = vi.sample(sample_key, state.parameters, num_samples=10000)
    
    # Plot true vs approx samples
    true_samples = random.multivariate_normal(
        sample_key, jnp.array([10.0, 10.0]), jnp.eye(2) * 5.0, (10000,)
    )
    _, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    sns.histplot(
        x=true_samples[:, 0], y=true_samples[:, 1], color="blue", ax=axs[0]
    )
    sns.histplot(
        x=approx_samples[:, 0], y=approx_samples[:, 1], color="red", ax=axs[1]
    )
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
