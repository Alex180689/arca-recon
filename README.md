# ARCA Cascade Reconstruction

This repository contains a specialized pipeline for reconstructing **cascade events** (neutrino interactions) simulated for the **KM3NeT ARCA** detector. The pipeline uses a hybrid approach combining neural networks with rigorous analytical geometry (Poisson and Bernoulli likelihoods) to reconstruct the vertex position, event time, energy, and direction of the cascades.

## How the Pipeline Works (Step-by-Step)

The reconstruction algorithm processes each event sequentially through three main phases:

### 1. Vertex and Time Optimization (Neural Network `model.h5` + Causal Penalty)
The first goal is to find the physical origin `(x, y, z)` and the exact time `(t)` of the neutrino interaction.
- **Initialization**: The algorithm generates 10 random starting positions within the active detector volume.
- **Neural Network Likelihood**: For each position/time hypothesis, it uses the pre-trained `model.h5` neural network. This network evaluates the expected hit likelihoods based on geometrical features and **time residuals** (the difference between the actual hit time and the expected light arrival time from the vertex). 
- **Causal Penalty**: An analytical penalty is added for non-causal early hits. Since light travels at the group velocity in water, any PMT registering a hit *before* light could physically reach it from the assumed vertex heavily penalizes that hypothesis.
- **Optimization**: The best 3 starting points are passed to the `L-BFGS-B` local optimizer which finds the precise `(x, y, z, t)` that minimizes the combined Neural Network NLL and causal penalty.

### 2. Energy Estimation (Gradient Boosting Regressor)
Once the vertex is fixed, we estimate the energy of the cascade.
- **Model**: This step uses the `gb_energy_model.pkl` model, which is a classic Gradient Boosting Regressor (Decision Trees).
- **Features**: The model takes only two inputs:
  1. The total number of hit PMTs (occupancy).
  2. The depth (`z` coordinate) of the reconstructed vertex.
- **Output**: The model outputs the estimated energy (in $log_{10}(E)$).

### 3. Direction Reconstruction (Neural Network + Bernoulli Likelihood)
This is the core of the pipeline and is fully vectorized using TensorFlow. Once Vertex, Time, and Energy are known, we search for the direction the neutrino was traveling `(theta, phi)`.
- **The Neural Network (`model_lambda.h5`)**: This is a deep neural network that predicts the expected amount of light on a PMT.
  - **Inputs**: It takes purely geometrical features relative to the hypothesis: distance from vertex, PMT orientation angle (`cos_alpha`), photon incidence angle (`cos_gamma`), and logarithmic distance.
  - **Output**: It predicts $\lambda$, which is the expected number of photons (Poisson rate) if the cascade had energy $E_0$.
- **Scaling**: The predicted $\lambda$ is multiplied by the actual estimated energy and modified by the water attenuation factor to get $\mu_{signal}$. We then add the standard ocean K40 background noise ($\mu_{bkg}$) to get the total expected hit rate $\mu$.
- **Bernoulli Likelihood**: The Poisson rate $\mu$ is analytically converted into a Bernoulli probability ($p = 1 - e^{-\mu}$) which represents the probability of the PMT firing at least once. The Negative Log-Likelihood is then computed across all PMTs.
- **Grid Search**: The algorithm tests 400 evenly distributed directions using a Fibonacci Sphere. The direction with the lowest NLL wins. Finally, a 10x10 fine-grid search is performed around the winner to pinpoint the exact angles.

## Files Included

- `arca_cascade_reconstruction.py`: The main robust and optimized pipeline. It reads the `.npz` event files, runs the full reconstruction step-by-step, and outputs the residual metrics (Truth vs. Reconstructed).
- `events_100.npz`: A sample dataset containing 100 simulated ARCA events so you can test the code immediately.
- `model_lambda.h5` & `gb_energy_model.pkl`: Pre-trained models for light yield estimation and energy regression.

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy, SciPy, Scikit-Learn

## How to Run

Activate your environment and run the main script:

```bash
conda activate tf
python arca_cascade_reconstruction.py
```

The script will process the 100 sample events, print a side-by-side comparison of the Ground Truth and Reconstructed parameters for each event, and output a summary table of the angular and spatial residuals at the end.
