# Wave-PINNs-Seismic

Solving the 1D seismic wave equation using **Physics-Informed Neural Networks (PINNs)** in PyTorch.

---

## Problem Statement

This project trains a neural network to solve the 1D acoustic wave equation:

$$u_{tt} = v^2 \, u_{xx}$$

where $v = 2$ is the wave propagation velocity, $x \in [0, 1]$, and $t \in [0, 1]$.

### Boundary & Initial Conditions

| Condition | Expression |
|---|---|
| Left boundary | $u(0, t) = \cos(2\pi t)$ |
| Right boundary | $u(1, t) = \cos\!\left[2\pi\!\left(t - \tfrac{1}{2}\right)\right]$ |
| Initial displacement | $u(x, 0) = \cos(\pi x)$ |
| Initial velocity | $\partial_t u(x, 0) = 2\pi \sin(\pi x)$ |

The analytical solution is:

$$u(x, t) = \cos\!\left[2\pi\!\left(t - \frac{x}{v}\right)\right]$$

---

## Approach

A feed-forward neural network is trained by minimizing a composite loss that encodes the physics directly, without any labeled simulation data.

### Network Architecture

- **Input:** $(x, t)$
- **Hidden layers:** 3 × fully-connected layers with 25 neurons each, Tanh activations
- **Output:** scalar $u(x, t)$

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}} + \mathcal{L}_{\text{IC}}$$

- **$\mathcal{L}_{\text{PDE}}$** — mean squared residual of $u_{tt} - v^2 u_{xx}$ over 1,000 collocation points
- **$\mathcal{L}_{\text{BC}}$** — MSE against boundary conditions at $x=0$ and $x=1$
- **$\mathcal{L}_{\text{IC}}$** — MSE against initial displacement and the PDE constraint at $t=0$

All derivatives are computed via PyTorch `autograd`.

---

## Requirements

```
torch
matplotlib
deepxde        # imported but architecture is custom PyTorch
```

Install with:

```bash
pip install torch matplotlib deepxde
```

> **Note:** The code was originally developed in Google Colab. Running locally requires removing the `pip install deepxde` line at the top of the script (or moving it to your setup step).

---

## Usage

```bash
python wave_pinns_seismic.py
```

Training runs for **10,000 epochs** with the Adam optimizer (lr = 0.01). Progress is printed every 100 epochs:

```
epoch:0/10000 || loss:1.4231
epoch:100/10000 || loss:0.0873
...
```

After training, the script produces:

- A **side-by-side 3D surface plot** comparing the analytical solution and model predictions over the $(x, t)$ domain
- An **animated comparison** of the two solutions evolving through time (rendered as inline HTML in Jupyter/Colab, or saveable as `wave_animation.mp4`)

---

## Results

The PINN learns to approximate the traveling wave purely from the physics constraints. After convergence, the predicted surface closely matches the analytical solution $u(x,t) = \cos[2\pi(t - x/v)]$.

| | Exact Solution | PINN Prediction |
|---|---|---|
| Method | Analytical formula | Neural network + autograd |
| Training data | None (unsupervised) | Collocation points only |

---

## Project Structure

```
wave_pinns_seismic.py   # Full model definition, training loop, and visualization
README.md
```

---

## Key Concepts

**Physics-Informed Neural Networks (PINNs)** embed differential equations into the loss function, allowing the network to learn solutions to PDEs without requiring ground-truth data. The method was introduced by [Raissi et al. (2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125).

---

## License

MIT
