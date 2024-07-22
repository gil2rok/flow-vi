# Flow VI

Cooking up something with variational inference and normalizing flows :robot: :repeat: :cook:.

## Installation + Quick Start

Clone this repository and install the dependencies in your favorite virtual environment.

```bash
git clone https://github.com/gil2rok/flow-vi.git
cd flow_vi
pip install -r requirements.txt
```

Then run the example script to train a RealNVP normalizing flow to learn a 2D Gaussian distribution. 

```bash
python example.py
```

The example runs on both GPU and CPU with multiple cores (thanks JAX!) but is much faster on a GPU.

## Resources

### Relevant Papers:
- Disentangling Capacity and Optimization for Flow-Based Variational Inference Using Modern Accelerators (not publicly available yet, implements VI-JAX below)
- :star: [Advances in Black-Box VI: Normalizing Flows, Importance Weighting, and Optimization](https://arxiv.org/abs/2006.10343)
- [Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://arxiv.org/abs/2406.07423)
- [Combining Normalizing Flows with Quasi Monte Carlo](https://arxiv.org/pdf/2401.05934)
-  :star: [Quasi Monte Carlo Flows](https://ml.cs.uni-kl.de/publications/2018/NeurIPS18_BDL_Quasi_Monte_Carlo_Flows.pdf)

### Background Papers:
- :star: [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)
- [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762)

### Github Repositories:
- :star: [JAX-Flows](https://github.com/ChrisWaites/jax-flows) (inspiration for my normalizing flow implementation)
- [Distrax](https://github.com/google-deepmind/distrax)
- [VI-JAX](https://github.com/abhiagwl/vijax)