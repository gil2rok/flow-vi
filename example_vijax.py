import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jax import random
from vijax import models, recipes, vardists, utils

sns.set_theme(style="whitegrid")

# Get model
# model = models.NealsFunnel(2) 
model = models.Banana(2)
# model = models.QuickGaussian(jnp.array([10.0, 10.0]), jnp.linalg.cholesky(jnp.eye(2) * 5.0), 2) # 2D Normal(10, 5)

# Get variational distribution
flow_q = vardists.RealNVP(
    model.ndim, # dimension of the target distribution
    num_transformations=10, # number of coupled RealNVP layers where each coupling consist of two RealNVP layers that alternate btwn which half of the data is masked
    num_hidden_units=32, # number of neurons in each hidden layer of the scale and translate networks
    num_hidden_layers=2, # number of hidden layers in each scale and translate network
    params_init_scale=0.001,
)

# Initialize the parameters of the flow variational distribution
flow_w = flow_q.initial_params()

# Create an instance of the recipe
recipe = recipes.SimpleVI(
    maxiter=10_000,
    batchsize=512 * (2**10),
    step_schedule="decay",
)

# Run the recipe with a flow variational distribution
new_q, new_w, vi_rez = recipe.run(target=model, vardist=flow_q, params=flow_w)

# [modified VI-JAX dependency] Plot the ELBO and Wasserstein distance
wass_dist = [float(d["Wasserstein1"]) for d in vi_rez[4]]
xs = [i for i in range(len(vi_rez[4]))]
df = pd.DataFrame({"iterations": xs, "wass_dist": wass_dist})
sns.relplot(x="iterations", y="wass_dist", kind="line", data=df, aspect=2)
plt.yscale("log")
plt.savefig("vijax_wass_dist.png")

# Generate samples from the true distribution
num_eval_samples = 512 * 1024
true_key = random.key(0)
true_samples = model.reference_samples(num_eval_samples, key=true_key)

# Generate samples from the flow
approx_keys = random.split(random.key(0), num_eval_samples)
approx_samples = jax.vmap(new_q.sample, in_axes=(None, 0))(new_w, approx_keys)
# sample_fn = lambda key: flow_q.sample(new_w, key)
# approx_samples = jax.vmap(sample_fn)(approx_keys)

# Compute Wasserstein distance
print(f"Wasserstein distance: {utils.wass1(true_samples, approx_samples)}")

# Plot true vs approx samples
_, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
sns.scatterplot(x=true_samples[:, 0], y=true_samples[:, 1], color="blue", ax=axs[0])
sns.scatterplot(x=approx_samples[:, 0], y=approx_samples[:, 1], color="red", ax=axs[1])
axs[0].set_title("True samples")
axs[1].set_title("Approximated samples")
plt.savefig("vijax_density.png")
