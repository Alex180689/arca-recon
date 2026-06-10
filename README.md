# ARCA Cascade Reconstruction

This repository contains a specialized pipeline for reconstructing **cascade events** (neutrino interactions) simulated for the **KM3NeT ARCA** detector. The pipeline uses a hybrid approach combining neural networks with rigorous analytical geometry (Poisson and Bernoulli likelihoods) to reconstruct the vertex position, event time, energy, and direction of the cascades.

## How the Pipeline Works (Step-by-Step)

The reconstruction algorithm processes each event sequentially through three main phases:

---

### 1. Vertex and Time Optimization — `model.h5` + Causal Penalty

The first goal is to find the physical origin `(x, y, z)` and the exact time `(t)` of the neutrino interaction. During this phase, the neutrino **direction and energy are kept fixed** (direction is initialized from the geometry of the first hit; energy is seeded to the median log-energy of the events in the file).

**Initialization**
The algorithm selects the first 6 early hits spread across at least 3 distinct Detection Units (DUs). These are used to generate 10 candidate starting positions inside the convex hull of these hits using Farthest Point Sampling.

**Neural Network Likelihood** (`model.h5`)
For each position/time hypothesis `(x, y, z, t)`, the pre-trained `model.h5` network evaluates the expected light yield on each individual hit. The network takes 5 per-hit features:
  1. Distance from vertex to hit position
  2. `cos_gamma` — cosine of the angle between the fixed neutrino direction and the vertex→hit direction
  3. `cos_alpha` — cosine between the hit direction and the PMT orientation
  4. **Time residual** — the difference between the actual hit time and the expected arrival time from the vertex at light speed in water
  5. `log(1 + distance)` — log-distance

Its output is the predicted expected photon count `μ_hit` (after un-scaling via `expm1` and multiplying by energy and a calibration constant). The **Poisson log-likelihood** is then computed as `-sum(log(μ_hit + K40))`, summed over all recorded hits.

**Causal Penalty**
An analytical penalty is added on top of the likelihood. Since light travels at the group velocity in water, the vertex time `t` must be early enough to causally explain the selected early hits. Vertices that violate this constraint are penalized with a squared soft margin.

**Optimization**
The 10 random starts are pre-evaluated, and the best 3 are passed to the `L-BFGS-B` local optimizer (with analytical gradient via `tf.GradientTape`), which finds the precise `(x, y, z, t)` that minimizes the combined NLL + causal penalty.

---

### 2. Energy Estimation — Gradient Boosting Regressor (`gb_energy_model.pkl`)

Once the vertex is fixed, we estimate the energy of the cascade. The `gb_energy_model.pkl` model is a classic Gradient Boosting Regressor (Decision Trees). It takes **three inputs** derived from the reconstruction:

  1. `log10(n_hits)` — the log of the total number of PMT hits
  2. `z` — the depth of the reconstructed vertex
  3. `r_horizontal = sqrt(x² + y²)` — the horizontal distance of the vertex from the detector center

Its output is the estimated energy in `log10(E / GeV)`, clipped to a physically reasonable range.

---

### 3. Direction Reconstruction — `model_lambda.h5` + Bernoulli Likelihood

This is the core of the pipeline. Once vertex, time, and energy are fixed, the algorithm searches for the direction `(theta, phi)` the neutrino was traveling. Unlike the vertex step, this phase uses the **aggregate per-PMT occupancy** `P_SIGNAL` (the probability that each PMT fires at least once), rather than individual hit times.

**The Neural Network** (`model_lambda.h5`)
This network predicts the expected Poisson photon rate `λ` on a given PMT for a given assumed direction. It takes 4 purely geometrical, direction-dependent features:
  1. Distance from vertex to PMT
  2. `cos_gamma` — cosine of the angle between the **candidate** direction and the vertex→PMT direction (this changes for each direction tested)
  3. `cos_alpha` — cosine of the angle between the vertex→PMT direction and the PMT orientation
  4. `log(1 + distance)` — log-distance

The network output is `log10(λ)`, so the actual rate is `λ = 10^y`. This is then scaled by the estimated energy and water attenuation: `μ_signal = λ × E × K_calibration × K_occupancy × attenuation`.

**Bernoulli Likelihood**
The total expected rate is `μ = μ_signal + K40_background`. This Poisson rate is analytically converted into the probability that the PMT fires at least once: `p_hit = 1 − e^(−μ)`. The **Bernoulli log-likelihood** is then computed across all PMTs: `LL = sum[ P_SIGNAL × log(p_hit) − (1 − P_SIGNAL) × μ ]`.

**Grid Search with TF Compilation**
- **Coarse step**: 400 directions uniformly distributed on the sphere (Fibonacci lattice) are evaluated. The NLL function is compiled once as a `@tf.function` and called per direction, avoiding Python-level retracing overhead.
- **Fine step**: A 10×10 grid is centered on the best coarse direction within a ±5° window, and the same compiled function is applied to all 100 fine points.

The direction with the lowest NLL among the fine grid points is the final reconstruction.

---

## Files Included

- `arca_cascade_reconstruction.py`: The main optimized pipeline. It reads the `.npz` event files, runs the full reconstruction step-by-step, and outputs the residual metrics (Truth vs. Reconstructed).
- `events_100.npz`: A sample dataset containing 100 simulated ARCA events so you can test the code immediately.
- `model.h5` — Neural network for hit-time likelihood (used in vertex optimization).
- `model_lambda.h5` — Neural network for PMT light yield estimation (used in direction search).
- `gb_energy_model.pkl` — Gradient Boosting model for energy estimation.
- `scalers.pkl` / `scalers_lambda.pkl` — StandardScaler objects for each network's input features.

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy, SciPy, Scikit-Learn, Matplotlib

## How to Run

Activate your environment and run the main script:

```bash
conda activate tf
python arca_cascade_reconstruction.py
```

The script will process the events, print a side-by-side comparison of the Ground Truth and Reconstructed parameters for each event, and output a summary table of the angular and spatial residuals at the end.
