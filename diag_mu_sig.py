"""
Diagnostic: direction reconstruction sweep over log10(lam) PMT thresholds.

Vectorized like def_recon_models:
  - geometry pre-computed once per event
  - all Fibonacci directions batched in a single model_lambda call
  - fine grid also batched

For each of the first 20 events:
  1. Compute lam per PMT at ground-truth vertex, cos_gamma=0.
  2. For each threshold t in {-8, -7, ..., 2}:
       - Keep only PMTs where log10(lam) >= t
       - Run Fibonacci coarse + fine grid search with Bernoulli likelihood
         (single batched model call per grid, with actual cos_gamma per direction)
       - Compute angular error vs ground truth
"""

import pickle
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE      = "events_100.npz"
N_EVENTS        = 20
FIBONACCI_N     = 400
COARSE_GRID_DEG = 10.0
CALIBRATION_K   = 2.697575011e-05
OCCUPANCY_SCALE = 0.655493
K40             = np.float32(1e-3)
THRESHOLDS      = list(range(-8, 3))   # -8, -7, ..., 2
CHUNK_SIZE      = 100                  # directions per chunk (limits peak RAM)

# ── Load model & scaler ───────────────────────────────────────────────────────
print("Loading model_lambda...")
model_lambda = tf.keras.models.load_model("model_lambda.h5", compile=False)
with open("scalers_lambda.pkl", "rb") as f:
    sc_lam = pickle.load(f)
sl_mean_np  = sc_lam.mean_.astype(np.float32)
sl_scale_np = sc_lam.scale_.astype(np.float32)

# Compile a @tf.function that accepts a 2-D batch of features
@tf.function(input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32)])
def run_lambda_batch(X_scaled):
    return tf.reshape(model_lambda(X_scaled, training=False), [-1])

# ── Load geometry & events ────────────────────────────────────────────────────
print("Loading data...")
with np.load(INPUT_FILE) as data:
    P_POS        = data["P_POS"].astype(np.float32)   # (N_pmt, 3)
    P_DIR        = data["P_DIR"].astype(np.float32)   # (N_pmt, 3)
    p_true_all   = data["p_true"]
    p_signal_all = data["P_SIGNAL"].astype(np.float32)

N_PMT = len(P_POS)

# ── Pre-build Fibonacci directions ────────────────────────────────────────────
def fibonacci_dirs(n):
    golden = (1 + np.sqrt(5)) / 2
    i      = np.arange(n, dtype=np.float64)
    y      = 1.0 - (i / (n - 1)) * 2.0
    r      = np.sqrt(np.maximum(1.0 - y*y, 0.0))
    tg     = 2.0 * np.pi * i / golden
    theta  = np.arccos(y)
    phi    = np.arctan2(np.sin(tg)*r, np.cos(tg)*r)
    dirs   = np.stack([np.sin(theta)*np.cos(phi),
                       np.sin(theta)*np.sin(phi),
                       np.cos(theta)], axis=1).astype(np.float32)
    return dirs, theta.astype(np.float64), phi.astype(np.float64)

fib_dirs, fib_theta, fib_phi = fibonacci_dirs(FIBONACCI_N)

def unit_from_angles(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)], dtype=np.float64)

def angular_error_deg(t1, p1, t2, p2):
    c = float(np.clip(np.dot(unit_from_angles(t1,p1), unit_from_angles(t2,p2)), -1, 1))
    return float(np.degrees(np.arccos(c)))


# ── Core: batched NLL over a grid of directions ───────────────────────────────
def nll_over_dirs(grid_dirs, ds, dp, ca, ld, at, obs, e_scale):
    """
    Evaluate Bernoulli NLL for every direction in grid_dirs (N_dirs, 3)
    using only the subset of PMTs supplied (already masked).

    Returns nll array of shape (N_dirs,).

    Strategy: build one big feature matrix (N_dirs * M, 4), single model call,
    reshape and reduce.  Done in CHUNK_SIZE direction chunks to bound RAM.
    """
    N_dirs = len(grid_dirs)
    M      = len(ds)

    # cos_gamma matrix: (M, N_dirs)
    cg_all = (dp @ grid_dirs.T)   # (M, N_dirs)

    nll_out = np.empty(N_dirs, dtype=np.float32)

    for start in range(0, N_dirs, CHUNK_SIZE):
        end    = min(start + CHUNK_SIZE, N_dirs)
        chunk  = end - start                         # number of dirs in this chunk
        cg_ch  = cg_all[:, start:end]                # (M, chunk)

        # Build feature matrix: (chunk * M, 4)
        # For each direction d, features are [dist, cg_d, ca, logd]
        # Tile static features along direction axis
        ds_rep = np.tile(ds, chunk)                  # (chunk * M,)  order: d0*M rows, d1*M rows...
        ca_rep = np.tile(ca, chunk)
        ld_rep = np.tile(ld, chunk)
        cg_rep = cg_ch.T.ravel()                     # (chunk * M,)  same order

        X_big  = np.stack([ds_rep, cg_rep, ca_rep, ld_rep], axis=1)   # (chunk*M, 4)
        X_sc   = (X_big - sl_mean_np) / sl_scale_np

        y_big  = run_lambda_batch(tf.constant(X_sc)).numpy()          # (chunk*M,)
        lam_ch = (10.0 ** y_big).reshape(chunk, M)                    # (chunk, M)

        mu_ch  = np.maximum(lam_ch * e_scale, 0.0) * at[None, :] + K40  # (chunk, M)
        ph_ch  = np.maximum(-np.expm1(-mu_ch), 1e-12)
        nll_out[start:end] = -np.sum(
            obs[None, :] * np.log(ph_ch) - (1.0 - obs[None, :]) * mu_ch,
            axis=1
        )

    return nll_out


def direction_search(mask, dist_safe, dir_p, ca_p, logd_safe, att, obs_hit, e_scale):
    """Coarse Fibonacci + fine 10×10 grid using only PMTs in mask."""
    ds  = dist_safe[mask].astype(np.float32)
    dp  = dir_p[mask].astype(np.float32)
    ca  = ca_p[mask].astype(np.float32)
    ld  = logd_safe[mask].astype(np.float32)
    at  = att[mask].astype(np.float32)
    obs = obs_hit[mask].astype(np.float32)

    # ── Coarse ───────────────────────────────────────────────────────────
    nll_c = nll_over_dirs(fib_dirs, ds, dp, ca, ld, at, obs, e_scale)
    bi    = int(np.argmin(nll_c))
    bc_t, bc_p = fib_theta[bi], fib_phi[bi]

    # ── Fine 10×10 ────────────────────────────────────────────────────────
    hc      = np.radians(COARSE_GRID_DEG / 2.0)
    th_fine = np.linspace(max(0.0, bc_t - hc), min(np.pi, bc_t + hc), 10)
    ph_fine = np.linspace(bc_p - hc, bc_p + hc, 10)
    th_g, ph_g = np.meshgrid(th_fine, ph_fine, indexing="ij")
    th_f, ph_f = th_g.ravel().astype(np.float64), ph_g.ravel().astype(np.float64)
    fine_dirs  = np.stack([np.sin(th_f)*np.cos(ph_f),
                           np.sin(th_f)*np.sin(ph_f),
                           np.cos(th_f)], axis=1).astype(np.float32)

    nll_f = nll_over_dirs(fine_dirs, ds, dp, ca, ld, at, obs, e_scale)
    fi    = int(np.argmin(nll_f))
    return float(th_f[fi]), float(ph_f[fi])


# ── Geometry helpers ──────────────────────────────────────────────────────────
def pmt_geometry(s_pos_np):
    v_p       = P_POS - s_pos_np
    dist_p    = np.maximum(np.linalg.norm(v_p, axis=1), 0.1)
    dist_safe = np.minimum(dist_p, 1000.0).astype(np.float32)
    dir_p     = (v_p / dist_p[:, np.newaxis]).astype(np.float32)
    logd_safe = np.log1p(dist_safe).astype(np.float32)
    ca_p      = (-np.sum(dir_p * P_DIR, axis=1)).astype(np.float32)
    att       = np.exp(-(dist_p - np.minimum(dist_p, 1000.0)) / 60.0).astype(np.float32)
    return dist_safe, dir_p, ca_p, logd_safe, att


def compute_log10_lam_cg0(dist_safe, ca_p, logd_safe):
    """Single model call with cos_gamma=0 for all PMTs."""
    cg0   = np.zeros(N_PMT, dtype=np.float32)
    X_lam = np.stack([dist_safe, cg0, ca_p, logd_safe], axis=1)
    X_sc  = (X_lam - sl_mean_np) / sl_scale_np
    y_lam = run_lambda_batch(tf.constant(X_sc)).numpy()
    return y_lam   # log10(lam) shape (N_pmt,)


# ── Main loop ─────────────────────────────────────────────────────────────────
import time

print(f"\nProcessing {N_EVENTS} events × {len(THRESHOLDS)} thresholds...\n")

results     = {t: [] for t in THRESHOLDS}
n_kept_log  = {t: [] for t in THRESHOLDS}
t0_global   = time.time()

for ev_idx in range(N_EVENTS):
    print(f'Evento {ev_idx+1}...')
    p_true    = p_true_all[ev_idx].astype(np.float64)
    s_pos     = p_true[:3].astype(np.float32)
    energy    = float(10.0 ** p_true[4])
    e_scale   = np.float32(energy * CALIBRATION_K * OCCUPANCY_SCALE)
    obs_hit   = np.clip(p_signal_all[ev_idx], 0.0, 1.0).astype(np.float32)
    true_t, true_p = float(p_true[5]), float(p_true[6])

    t0_ev = time.time()

    # --- Step 1: geometry + lam with cos_gamma=0 (one model call) -----------
    dist_safe, dir_p, ca_p, logd_safe, att = pmt_geometry(s_pos)
    log10_lam = compute_log10_lam_cg0(dist_safe, ca_p, logd_safe)

    print(f"Event {ev_idx+1:>2}  logE={p_true[4]:.2f}", flush=True)

    # --- Step 2: threshold sweep --------------------------------------------
    for t in THRESHOLDS:
        mask   = log10_lam >= t
        n_kept = int(mask.sum())
        n_kept_log[t].append(n_kept)

        if n_kept < 10:
            results[t].append(np.nan)
            print(f"    thr={t:>3d}  kept={n_kept:>5}  -> skip (too few PMTs)")
            continue

        reco_t, reco_p = direction_search(mask, dist_safe, dir_p,
                                          ca_p, logd_safe, att, obs_hit, e_scale)
        angle = angular_error_deg(true_t, true_p, reco_t, reco_p)
        results[t].append(angle)
        print(f"    thr={t:>3d}  kept={n_kept:>5}  angle={angle:.2f}°", flush=True)

    print(f"  → event time: {time.time()-t0_ev:.1f}s\n")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  {'Threshold':<12} | {'N_pmt med':<10} | {'Mean°':<8} | "
      f"{'Median°':<8} | {'P90°':<8} | {'Max°':<8} | {'N ok':<5}")
print("-"*70)
for t in THRESHOLDS:
    angles = np.array([a for a in results[t] if not np.isnan(a)])
    n_med  = int(np.median(n_kept_log[t])) if n_kept_log[t] else 0
    n_ok   = len(angles)
    if n_ok == 0:
        print(f"  {t:<12} | {n_med:<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {n_ok:<5}")
    else:
        print(f"  {t:<12} | {n_med:<10} | {np.mean(angles):<8.3f} | "
              f"{np.median(angles):<8.3f} | {np.percentile(angles,90):<8.3f} | "
              f"{np.max(angles):<8.3f} | {n_ok:<5}")
print("="*70)
print(f"\nTotal time: {time.time()-t0_global:.1f}s")
