# ARCA Cascade Reconstruction

This repository contains a specialized pipeline for reconstructing **cascade events** (neutrino interactions) simulated for the **KM3NeT ARCA** detector. The pipeline uses a hybrid approach combining neural networks with rigorous analytical geometry (Poisson and Bernoulli likelihoods) to reconstruct the vertex position, event time, energy, and direction of the cascades.

## Features

- **Vertex and Time Reconstruction**: Uses an early-hit causal model with multi-start local optimization (L-BFGS-B).
- **Direction Reconstruction**: Vectorized grid search using a 400-point Fibonacci sphere followed by a fine 10x10 grid refinement.
- **Energy Estimation**: Uses a Gradient Boosting regressor based on PMT hit counts and vertex depth.
- **Neural Network Likelihoods**: Employs a pre-trained neural network (`model_lambda.h5`) to estimate the expected light yield (Poisson rate) on individual PMTs based on distance, PMT orientation, and attenuation.

## Files Included

- `def_recon_models.py`: The main robust and optimized pipeline. It reads the `.npz` event files, runs the full reconstruction step-by-step, and outputs the residual metrics (Truth vs. Reconstructed).
- `diag_mu_sig.py`: A diagnostic script to evaluate the PMT light yield probabilities (`lam` and `mu_sig`) directly from the Neural Network, without running the full reconstruction. Useful for threshold sweeps and debugging.
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
python def_recon_models.py
```

The script will process the 100 sample events, print a side-by-side comparison of the Ground Truth and Reconstructed parameters for each event, and output a summary table of the angular and spatial residuals at the end.
