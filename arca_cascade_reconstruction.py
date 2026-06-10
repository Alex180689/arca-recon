"""
KM3NeT cascade reconstruction pipeline - single file, TF optimized.

Main optimizations over the previous version:
  - Angular grid search (coarse + fine) fully batched: a single
    call to the lambda model for all grid points instead of
    400+100 separate calls in a Python loop.
  - Pre-computation of PMT geometry (distances, directions, log-distances)
    which remains fixed during the angular grid search: it is stored as a
    TF constant and reused for each event without recalculating.
  - Fibonacci sphere points pre-calculated once at startup
    as TF tensors (theta, phi, s_dir pre-built as an Nx3 matrix).
  - Fine grid: s_dir for all 100 points calculated in a single
    vectorized operation, no internal Python loop.
  - eval_nll for the vertex optimizer remains scalar (necessary for the
    L-BFGS-B gradient), but the hit geometry is already pre-loaded as a
    constant tensor before the starts loop.
"""

import csv
import gc
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ── Force CPU execution (no GPU) ─────────────────────────────────────────────
#tf.config.set_visible_devices([], 'GPU')
from scipy.optimize import minimize
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn.cluster import DBSCAN

# ── Run configuration ────────────────────────────────────────────────────────

INPUT_FILES = [
    "events_100.npz",
    "events_200.npz",
    "events_300_1.npz",
    "events_300_2.npz",
    "events_300_3.npz",
]

N_EVENTS      = 20
OUTPUT_PREFIX = "full_reco_gb"
RANDOM_SEED   = 12345

# ── Physical constants ───────────────────────────────────────────────────────

C_WATER        = 0.2222
SCALE_FACTOR   = 100.0
CALIBRATION_K  = 2.697575011e-05
K40_PE_PER_HIT = 1e-3

# ── Likelihood (binary occupancy) ────────────────────────────────────────────

OCCUPANCY_LIGHT_SCALE = 0.655493

# ── Vertex reconstruction ────────────────────────────────────────────────────

VERTEX_MAXITER         = 40
ENERGY_BOUNDS          = (3.5, 7.5)
N_VERTEX_RANDOM_STARTS = 10
MIN_VERTEX_DU          = 3
DU_CLUSTER_EPS_M       = 1.0
CAUSAL_PENALTY_WEIGHT  = 1.0
CAUSAL_PENALTY_TOL     = 20.0
CAUSAL_PENALTY_SCALE   = 10.0

# ── Direction reconstruction ─────────────────────────────────────────────────

COARSE_GRID_DEG     = 10.0
FIBONACCI_N_POINTS  = 400
GB_MODEL_PATH       = "gb_energy_model.pkl"
BATCH_CHUNK_SIZE    = 50    # directions per chunk in grid search (limits RAM)
N_VERTEX_TOP_STARTS = 3     # how many starts to optimize after batch pre-filtering

# ── Global variables (models, scalers, geometry) ─────────────────────────────

model_hits   = None
model_lambda = None
sh_mean = sh_std = None
sl_mean = sl_scale = None

# Detector geometry tensors (loaded once, reused for all events)
P_POS_tf = None   # (N_pmt, 3)  float64
P_DIR_tf = None   # (N_pmt, 3)  float64
P_POS    = None   # numpy, for DBSCAN/cKDTree
P_DIR    = None   # numpy, for reference

# Pre-calculated once at startup
_fib_dirs_tf  = None   # (FIBONACCI_N_POINTS, 3) float64 - Fibonacci directions
_fib_theta_np = None   # (FIBONACCI_N_POINTS,)   numpy   - theta, for fine grid
_fib_phi_np   = None   # (FIBONACCI_N_POINTS,)   numpy   - phi,   for fine grid


# ════════════════════════════════════════════════════════════════════════════
# I/O and setup
# ════════════════════════════════════════════════════════════════════════════

def load_events_from_npz(npz_path):
    """Loads a .npz archive and returns a list of event dictionaries."""
    global P_POS, P_DIR, P_POS_tf, P_DIR_tf
    with np.load(npz_path) as data:
        P_POS = data["P_POS"].astype(np.float64)
        P_DIR = data["P_DIR"].astype(np.float64)
        P_POS_tf = tf.constant(P_POS, dtype=tf.float64)
        P_DIR_tf = tf.constant(P_DIR, dtype=tf.float64)
        p_signal_all = data["P_SIGNAL"].astype(np.float64)
        offsets  = data["hit_offsets"]
        n_events = len(data["p_true"])
        events   = []
        for i in range(n_events):
            s, e = offsets[i], offsets[i + 1]
            events.append({
                "p_true":        data["p_true"][i],
                "carica_totale": data["carica_totale"][i],
                "n_hits":        data["n_hits"][i],
                "H_POS":         data["hit_pos"][s:e].astype(np.float64),
                "H_DIR":         data["hit_dir"][s:e].astype(np.float64),
                "H_T":           data["hit_t"][s:e].astype(np.float64),
                "H_A":           data["hit_a"][s:e],
                "P_SIGNAL":      p_signal_all[i],
            })
    return events


def load_models():
    """Loads Keras models, scalers, and pre-calculates Fibonacci tensors."""
    global model_hits, model_lambda, sh_mean, sh_std, sl_mean, sl_scale
    global _fib_dirs_tf, _fib_theta_np, _fib_phi_np

    model_hits = tf.keras.models.load_model("model.h5", compile=False)
    with open("scalers.pkl", "rb") as f:
        sc_hits = pickle.load(f)

    model_lambda = tf.keras.models.load_model("model_lambda.h5", compile=False)
    with open("scalers_lambda.pkl", "rb") as f:
        sc_lam = pickle.load(f)

    sh_mean = tf.constant(
        [sc_hits["dist"]["mean"], 0.0, 0.0,
         sc_hits["time"]["mean"], sc_hits["log_dist"]["mean"]],
        dtype=tf.float32,
    )
    sh_std = tf.constant(
        [sc_hits["dist"]["std"], 1.0, 1.0,
         sc_hits["time"]["std"], sc_hits["log_dist"]["std"]],
        dtype=tf.float32,
    )
    sl_mean  = tf.constant(sc_lam.mean_,  dtype=tf.float32)
    sl_scale = tf.constant(sc_lam.scale_, dtype=tf.float32)

    # Pre-calculate Fibonacci points as a TF tensor once
    theta_np, phi_np = _fibonacci_sphere_arrays(FIBONACCI_N_POINTS)
    _fib_theta_np = theta_np
    _fib_phi_np   = phi_np
    # (N, 3) matrix of Fibonacci unit directions
    dirs = np.stack([
        np.sin(theta_np) * np.cos(phi_np),
        np.sin(theta_np) * np.sin(phi_np),
        np.cos(theta_np),
    ], axis=1)  # (N, 3)
    _fib_dirs_tf = tf.constant(dirs, dtype=tf.float64)


def _fibonacci_sphere_arrays(n_samples):
    """Returns (theta, phi) as numpy arrays for n_samples Fibonacci points."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    i     = np.arange(n_samples, dtype=np.float64)
    y     = 1.0 - (i / (n_samples - 1)) * 2.0
    r     = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    theta_g = 2.0 * np.pi * i / golden_ratio
    theta = np.arccos(y)
    phi   = np.arctan2(np.sin(theta_g) * r, np.cos(theta_g) * r)
    return theta, phi


def event_tensors(event):
    """Returns the TF tensors for an event."""
    return (
        tf.constant(event["H_POS"].reshape(-1, 3), dtype=tf.float64),
        tf.constant(event["H_DIR"].reshape(-1, 3), dtype=tf.float64),
        tf.constant(event["H_T"].reshape(-1),       dtype=tf.float64),
        tf.constant(event["P_SIGNAL"].reshape(-1),  dtype=tf.float64),
    )


# tf.Variable for pre-calculated geometry (initialized to None, created on first use)
_geom_vars_initialized = False
_geom_dist_safe = None
_geom_logd      = None
_geom_ca_p      = None
_geom_att       = None
_geom_obs_hit   = None
_geom_dir_p     = None
_geom_e_scale   = None
# Reference to the compiled @tf.function (created after var init)
_nll_fn         = None


def _build_nll_function():
    """Creates the @tf.function after the tf.Variable have been initialized."""
    @tf.function(input_signature=[tf.TensorSpec(shape=(3,), dtype=tf.float32)])
    def nll_single_dir(s_dir):
        cg_p = tf.linalg.matvec(_geom_dir_p, s_dir)
        X_lam = tf.stack([_geom_dist_safe, cg_p, _geom_ca_p, _geom_logd], axis=1)
        X_scaled = (X_lam - sl_mean) / sl_scale
        y_lam = tf.reshape(model_lambda(X_scaled, training=False), [-1])
        # The model predicts log10(lambda), so we must use 10**y_lam
        lam = tf.pow(10.0, y_lam)
        mu_sig = tf.maximum(lam * _geom_e_scale, 0.0) * _geom_att
        mu     = mu_sig + tf.constant(K40_PE_PER_HIT, dtype=tf.float32)
        p_hit  = tf.maximum(-tf.math.expm1(-mu), tf.constant(1e-12, dtype=tf.float32))
        ll = _geom_obs_hit * tf.math.log(p_hit) - (1.0 - _geom_obs_hit) * mu
        return -tf.reduce_sum(ll)
    return nll_single_dir


def tf_nll_batch_dirs(s_dirs_all_f32):
    """Evaluates NLL for K directions. Geometry is already in tf.Variable."""
    K = s_dirs_all_f32.shape[0]
    results = np.empty(K, dtype=np.float32)
    for i in range(K):
        results[i] = _nll_fn(s_dirs_all_f32[i]).numpy()
    return results


def set_pmt_geometry(s_pos_np, energy_f64, P_SIGNAL_np):
    """
    Pre-calculates PMT geometry and writes it to the global tf.Variable.
    On first use, it creates the Variables at the correct size.
    """
    global _geom_vars_initialized, _nll_fn
    global _geom_dist_safe, _geom_logd, _geom_ca_p, _geom_att
    global _geom_obs_hit, _geom_dir_p, _geom_e_scale

    v_p       = P_POS - s_pos_np
    dist_p    = np.maximum(np.linalg.norm(v_p, axis=1), 0.1)
    dist_safe = np.minimum(dist_p, 1000.0).astype(np.float32)
    dir_p     = (v_p / dist_p[:, np.newaxis]).astype(np.float32)
    logd_safe = np.log1p(dist_safe).astype(np.float32)
    ca_p      = (-np.sum(dir_p * P_DIR, axis=1)).astype(np.float32)
    att       = np.exp(-(dist_p - np.minimum(dist_p, 1000.0)) / 60.0).astype(np.float32)
    e_scale   = np.float32(energy_f64 * CALIBRATION_K * OCCUPANCY_LIGHT_SCALE)
    obs_hit   = np.clip(P_SIGNAL_np, 0.0, 1.0).astype(np.float32)

    if not _geom_vars_initialized:
        _geom_dist_safe = tf.Variable(dist_safe, trainable=False)
        _geom_logd      = tf.Variable(logd_safe, trainable=False)
        _geom_ca_p      = tf.Variable(ca_p, trainable=False)
        _geom_att       = tf.Variable(att, trainable=False)
        _geom_obs_hit   = tf.Variable(obs_hit, trainable=False)
        _geom_dir_p     = tf.Variable(dir_p, trainable=False)
        _geom_e_scale   = tf.Variable(e_scale, trainable=False)
        _nll_fn = _build_nll_function()
        _geom_vars_initialized = True
    else:
        _geom_dist_safe.assign(dist_safe)
        _geom_logd.assign(logd_safe)
        _geom_ca_p.assign(ca_p)
        _geom_att.assign(att)
        _geom_obs_hit.assign(obs_hit)
        _geom_dir_p.assign(dir_p)
        _geom_e_scale.assign(e_scale)


# ════════════════════════════════════════════════════════════════════════════
# Scalar Likelihood (for L-BFGS-B vertex optimizer, requires gradient)
# ════════════════════════════════════════════════════════════════════════════

@tf.function
def vertex_time_nll_tf(x, fixed_theta_phi_loge, H_POS, H_DIR, H_T, early_pos, early_t, causal_weight):
    """NLL for vertex reconstruction (hits model + causal penalty)."""
    s_pos = x[:3]; st = x[3]
    theta, phi, log_e = (fixed_theta_phi_loge[0],
                         fixed_theta_phi_loge[1],
                         fixed_theta_phi_loge[2])
    s_dir  = tf.stack([tf.sin(theta) * tf.cos(phi),
                       tf.sin(theta) * tf.sin(phi),
                       tf.cos(theta)])
    energy = tf.pow(tf.constant(10.0, dtype=tf.float64), log_e)

    v_h         = H_POS - s_pos
    dist_h      = tf.maximum(tf.norm(v_h, axis=1), 0.1)
    dir_h       = v_h / tf.expand_dims(dist_h, -1)
    cos_gamma_h = tf.reduce_sum(s_dir * dir_h, axis=1)
    cos_alpha_h = -tf.reduce_sum(dir_h * H_DIR, axis=1)
    time_res_h  = H_T - st - dist_h / C_WATER
    logd_h      = tf.math.log1p(dist_h)

    X_hits   = tf.stack([dist_h, cos_gamma_h, cos_alpha_h, time_res_h, logd_h], axis=1)
    X32_hits = tf.cast((tf.cast(X_hits, tf.float32) - sh_mean) / sh_std, tf.float32)
    y_hits   = tf.cast(model_hits(X32_hits, training=False), tf.float64)
    mu_hit   = tf.maximum(
        tf.reshape((tf.math.expm1(y_hits) / SCALE_FACTOR) * energy * CALIBRATION_K, [-1]), 0.0)
    nll = -tf.reduce_sum(tf.math.log(mu_hit + K40_PE_PER_HIT))

    early_dist        = tf.maximum(tf.norm(early_pos - s_pos, axis=1), 0.1)
    latest_allowed_t0 = early_t - early_dist / C_WATER
    violation         = st - latest_allowed_t0 - CAUSAL_PENALTY_TOL
    causal_penalty    = causal_weight * tf.reduce_mean(
        tf.square(tf.nn.relu(violation / CAUSAL_PENALTY_SCALE)))
    return nll + causal_penalty


# ════════════════════════════════════════════════════════════════════════════
# Vertex and time reconstruction
# ════════════════════════════════════════════════════════════════════════════

def build_du_lookup():
    labels = DBSCAN(eps=DU_CLUSTER_EPS_M, min_samples=10).fit_predict(P_POS[:, :2])
    if np.any(labels < 0):
        raise RuntimeError("DBSCAN left PMTs without DU: increase DU_CLUSTER_EPS_M")
    return cKDTree(P_POS), labels.astype(np.int32)


def unit_from_angles(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def angles_from_unit(vec):
    norm = np.linalg.norm(vec)
    if norm == 0.0 or not np.isfinite(norm):
        return np.pi / 2.0, 0.0
    vec   = vec / norm
    theta = np.arccos(np.clip(vec[2], -1.0, 1.0))
    phi   = np.arctan2(vec[1], vec[0])
    return float(theta), float(phi)


def initial_vertex_and_direction(event):
    hit_pos   = event["H_POS"].reshape(-1, 3)
    hit_t     = event["H_T"].reshape(-1)
    first_idx = int(np.argmin(hit_t))
    centroid  = np.mean(hit_pos, axis=0)
    theta0, phi0 = angles_from_unit(centroid - hit_pos[first_idx])
    t0 = float(hit_t[first_idx]) - 50.0
    return hit_pos[first_idx], t0, theta0, phi0


def select_early_hits_on_multiple_dus(event, pmt_tree, du_labels):
    hit_pos    = event["H_POS"].reshape(-1, 3)
    hit_t      = event["H_T"].reshape(-1)
    _, pmt_idx = pmt_tree.query(hit_pos, k=1)
    hit_du     = du_labels[pmt_idx]
    order      = np.argsort(hit_t)
    selected, used_du = [], set()
    for idx in order:
        du = int(hit_du[idx])
        if du not in used_du:
            selected.append(int(idx)); used_du.add(du)
        if len(used_du) >= MIN_VERTEX_DU:
            break
    for idx in order:
        idx = int(idx)
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= 6:
            break
    selected = np.asarray(selected[:6], dtype=np.int32)
    return hit_pos[selected], hit_t[selected], hit_du[selected]


def initial_t0_from_position(pos, sel_pos, sel_t):
    distances = np.linalg.norm(sel_pos - pos[None, :], axis=1)
    return float(np.median(sel_t - distances / C_WATER))


def sample_farthest_points_in_hull(sel_pos, rng, n_samples, n_candidates=2000):
    p_min, p_max = np.min(sel_pos, axis=0), np.max(sel_pos, axis=0)
    try:
        delaunay = Delaunay(sel_pos)
        pool     = rng.uniform(p_min, p_max, size=(n_candidates * 10, 3))
        inside   = pool[delaunay.find_simplex(pool) >= 0]
        if len(inside) < n_samples:
            return rng.uniform(p_min, p_max, size=(n_samples, 3)), "bounding_box"
    except QhullError:
        return rng.uniform(p_min, p_max, size=(n_samples, 3)), "bounding_box"
    centroid     = inside.mean(axis=0)
    first_idx    = int(np.argmax(np.linalg.norm(inside - centroid, axis=1)))
    selected_idx = [first_idx]
    min_dists    = np.linalg.norm(inside - inside[first_idx], axis=1)
    for _ in range(n_samples - 1):
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)
        min_dists = np.minimum(min_dists, np.linalg.norm(inside - inside[next_idx], axis=1))
    return inside[selected_idx], "convex_hull_fps"


def make_random_vertex_starts(event, pmt_tree, du_labels, rng):
    sel_pos, sel_t, sel_du = select_early_hits_on_multiple_dus(event, pmt_tree, du_labels)
    random_positions, sampling_method = sample_farthest_points_in_hull(
        sel_pos, rng, N_VERTEX_RANDOM_STARTS)
    starts = [
        np.array([pos[0], pos[1], pos[2],
                  initial_t0_from_position(pos, sel_pos, sel_t)], dtype=np.float64)
        for pos in random_positions
    ]
    return starts, sampling_method, int(len(set(map(int, sel_du)))), sel_pos, sel_t


def reconstruct_vertex_time(event, h_pos, h_dir, h_t, loge_seed, pmt_tree, du_labels, rng):
    _, _, theta0, phi0 = initial_vertex_and_direction(event)
    fixed = tf.constant([theta0, phi0, loge_seed], dtype=tf.float64)
    p_min = P_POS.min(axis=0) - 100.0
    p_max = P_POS.max(axis=0) + 100.0
    min_hit_t = float(event["H_T"].min())
    bounds = [(float(p_min[0]), float(p_max[0])),
              (float(p_min[1]), float(p_max[1])),
              (float(p_min[2]), float(p_max[2])),
              (-200.0, min_hit_t + 20.0)]

    (starts, sampling_method,
     selected_du, sel_pos, sel_t) = make_random_vertex_starts(event, pmt_tree, du_labels, rng)
    early_pos_tf     = tf.constant(sel_pos, dtype=tf.float64)
    early_t_tf       = tf.constant(sel_t,   dtype=tf.float64)
    causal_weight_tf = tf.constant(CAUSAL_PENALTY_WEIGHT, dtype=tf.float64)

    @tf.function(input_signature=[tf.TensorSpec(shape=(4,), dtype=tf.float64)])
    def value_and_grad(x_tf):
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            value = vertex_time_nll_tf(x_tf, fixed, h_pos, h_dir, h_t,
                                       early_pos_tf, early_t_tf, causal_weight_tf)
        return value, tape.gradient(value, x_tf)

    starts_clipped = np.array([
        [np.clip(s[i], bounds[i][0], bounds[i][1]) for i in range(4)]
        for s in starts
    ], dtype=np.float64)  # (N_starts, 4)

    start_nlls = np.array([
        float(vertex_time_nll_tf(
            tf.constant(s, dtype=tf.float64), fixed, h_pos, h_dir, h_t,
            early_pos_tf, early_t_tf, causal_weight_tf).numpy())
        for s in starts_clipped
    ])
    # Select the top-N_VERTEX_TOP_STARTS starts with the lowest NLL
    top_idx = np.argsort(start_nlls)[:N_VERTEX_TOP_STARTS]

    def scipy_fun(x_np):
        v, g = value_and_grad(tf.constant(x_np, dtype=tf.float64))
        return float(v.numpy()), g.numpy().astype(np.float64)

    best = None
    for idx in top_idx:
        result = minimize(scipy_fun, starts_clipped[idx], method="L-BFGS-B", jac=True,
                          bounds=bounds,
                          options={"maxiter": VERTEX_MAXITER, "ftol": 1e-8, "gtol": 1e-5})
        if best is None or result.fun < best.fun:
            best = result

    return (best.x.astype(np.float64), theta0, phi0,
            bool(best.success), str(best.message),
            sampling_method, selected_du, float(best.fun))


# ════════════════════════════════════════════════════════════════════════════
# Energy estimation (Gradient Boosting)
# ════════════════════════════════════════════════════════════════════════════

def gb_energy_guess(gb_model, vertex_time, n_hits):
    log_nhits = np.log10(max(float(n_hits), 1.0))
    reco_z    = float(vertex_time[2])
    r_vertex  = float(np.sqrt(vertex_time[0] ** 2 + vertex_time[1] ** 2))
    loge      = gb_model.predict([[log_nhits, reco_z, r_vertex]])[0]
    return float(np.clip(loge, *ENERGY_BOUNDS))


# ════════════════════════════════════════════════════════════════════════════
# Direction reconstruction - batched grid search
# ════════════════════════════════════════════════════════════════════════════

def run_angle_search(params_init, P_SIGNAL_np):
    """
    Angular search in two steps, fully batched in TF:
      1) Coarse: chunked on FIBONACCI_N_POINTS points
      2) Fine:   chunked on 10x10 sub-grid points
    Position, time and energy are fixed to params_init.
    P_SIGNAL_np is a numpy array (N_pmt,).
    """
    s_pos_np  = params_init[:3]
    energy_f64 = 10.0 ** params_init[4]

    # Write PMT geometry to global tf.Variables
    set_pmt_geometry(s_pos_np, energy_f64, P_SIGNAL_np)

    # Fibonacci directions as float32
    fib_dirs_f32 = tf.constant(_fib_dirs_tf.numpy().astype(np.float32))

    # ── Step 1: Coarse Fibonacci ─────────────────────────────────────────
    nll_coarse = tf_nll_batch_dirs(fib_dirs_f32)  # (FIBONACCI_N_POINTS,)

    best_idx         = int(np.argmin(nll_coarse))
    best_coarse_theta = _fib_theta_np[best_idx]
    best_coarse_phi   = _fib_phi_np[best_idx]

    # ── Step 2: Fine grid 10x10 ──────────────────────────────────────────
    half_cell       = np.radians(COARSE_GRID_DEG / 2.0)
    theta_fine      = np.linspace(max(0.0, best_coarse_theta - half_cell),
                                   min(np.pi, best_coarse_theta + half_cell), 10)
    phi_fine        = np.linspace(best_coarse_phi - half_cell,
                                   best_coarse_phi + half_cell, 10)
    th_grid, ph_grid = np.meshgrid(theta_fine, phi_fine, indexing="ij")
    th_flat = th_grid.ravel()
    ph_flat = ph_grid.ravel()
    fine_dirs = np.stack([
        np.sin(th_flat) * np.cos(ph_flat),
        np.sin(th_flat) * np.sin(ph_flat),
        np.cos(th_flat),
    ], axis=1).astype(np.float32)  # (100, 3)
    fine_dirs_tf = tf.constant(fine_dirs)

    nll_fine = tf_nll_batch_dirs(fine_dirs_tf)  # (100,)

    best_fine_idx   = int(np.argmin(nll_fine))
    best_fine_theta = float(th_flat[best_fine_idx])
    best_fine_phi   = float(ph_flat[best_fine_idx])

    p_coarse    = np.array(params_init, dtype=np.float64, copy=True)
    p_coarse[5] = best_coarse_theta
    p_coarse[6] = best_coarse_phi

    p_fine    = np.array(params_init, dtype=np.float64, copy=True)
    p_fine[5] = best_fine_theta
    p_fine[6] = best_fine_phi

    dir_true  = unit_from_angles(params_init[5], params_init[6])
    dir_pred  = unit_from_angles(best_fine_theta, best_fine_phi)
    cos_alpha = float(np.clip(np.dot(dir_true, dir_pred), -1.0, 1.0))

    return {
        "coarse_params":   p_coarse,
        "fine_params":     p_fine,
        "space_angle_deg": float(np.degrees(np.arccos(cos_alpha))),
    }


# ════════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════════

def angular_error_deg(true_theta, true_phi, reco_theta, reco_phi):
    return float(np.degrees(np.arccos(
        np.clip(np.dot(unit_from_angles(true_theta, true_phi),
                       unit_from_angles(reco_theta, reco_phi)), -1.0, 1.0))))


def save_error_plots(errors):
    plot_specs = [
        ("dx",    "X residual [m]",      "X residual distribution"),
        ("dy",    "Y residual [m]",      "Y residual distribution"),
        ("dz",    "Z residual [m]",      "Z residual distribution"),
        ("dt",    "T residual",          "Time residual distribution"),
        ("angle", "Angular error [deg]", "Direction angular error distribution"),
        ("dlogE", "log10(E) residual",   "log10(E) residual distribution"),
    ]
    for key, xlabel, title in plot_specs:
        values = np.asarray(errors[key], dtype=np.float64)
        plt.figure(figsize=(7.5, 5.0))
        plt.hist(values, bins=30, alpha=0.75, color="#4c78a8")
        plt.axvline(np.median(values), color="#f58518", linestyle="--",
                    linewidth=2.0, label="median")
        plt.xlabel(xlabel); plt.ylabel("Events"); plt.title(title)
        plt.grid(axis="y", alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(f"{OUTPUT_PREFIX}_{key}_distribution.png", dpi=180)
        plt.close()


def summarize(name, values):
    values = np.asarray(values, dtype=np.float64)
    return (f"{name:<15} | {np.mean(values):>8.3f} | {np.median(values):>8.3f} | "
            f"{np.percentile(np.abs(values), 90):>8.3f} | {np.max(np.abs(values)):>8.3f}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    load_models()

    with open(GB_MODEL_PATH, "rb") as f:
        gb_model = pickle.load(f)
    print(f"GB model loaded from {GB_MODEL_PATH}")

    rng    = np.random.default_rng(RANDOM_SEED)
    rows   = []
    errors = {"dx": [], "dy": [], "dz": [], "dt": [], "angle": [], "dlogE": []}
    t0_global = time.time()
    events_remaining = N_EVENTS  # global budget, consumed across files

    for input_file in INPUT_FILES:
        if events_remaining <= 0:
            break

        print(f"\n{'='*60}\nProcessing: {input_file}\n{'='*60}")

        # Take only as many events as still needed from this file
        events = load_events_from_npz(input_file)[:events_remaining]
        events_remaining -= len(events)
        pmt_tree, du_labels = build_du_lookup()
        seed_loge = float(np.median([ev["p_true"][4] for ev in events]))
        print(f"  {len(events)} events from this file  |  seed logE={seed_loge:.3f}")

        # Print header for each event log
        print(f"{'Event':>5} | "
              f"{'Ground Truth (x, y, z, t, E, theta, phi)':^55} | "
              f"{'Reconstructed (x, y, z, t, E, theta, phi)':^55} | "
              f"{'Time (s)':>8}")
        print("-" * 130)

        for idx, event in enumerate(events):
            event_t0 = time.time()
            p_true   = np.asarray(event["p_true"], dtype=np.float64)
            n_hits   = int(event["n_hits"])
            h_pos, h_dir, h_t, p_signal_tf = event_tensors(event)
            p_signal_np = event["P_SIGNAL"].reshape(-1).astype(np.float64)

            # 1. Vertex and time
            (vertex_time, theta0, phi0,
             vertex_ok, vertex_msg,
             sampling_method, selected_du,
             vertex_nll) = reconstruct_vertex_time(
                event, h_pos, h_dir, h_t, seed_loge, pmt_tree, du_labels, rng)


            # 2. Energy (GB)
            loge_gb = gb_energy_guess(gb_model, vertex_time, n_hits)

            # 3. Direction (batched)
            params_for_angle     = p_true.copy()
            params_for_angle[:3] = vertex_time[:3]
            params_for_angle[3]  = vertex_time[3]
            params_for_angle[4]  = loge_gb
            direction_result     = run_angle_search(params_for_angle, p_signal_np)
            direction_params     = direction_result["fine_params"]

            dx      = float(vertex_time[0] - p_true[0])
            dy      = float(vertex_time[1] - p_true[1])
            dz      = float(vertex_time[2] - p_true[2])
            dt      = float(vertex_time[3] - p_true[3])
            angle   = angular_error_deg(p_true[5], p_true[6],
                                        direction_params[5], direction_params[6])
            dloge   = float(loge_gb - p_true[4])
            elapsed = time.time() - event_t0

            for key, val in zip(["dx", "dy", "dz", "dt", "angle", "dlogE"],
                                 [dx, dy, dz, dt, angle, dloge]):
                errors[key].append(val)

            rows.append({
                "file": input_file, "event": idx + 1,
                "true_x": p_true[0], "true_y": p_true[1],
                "true_z": p_true[2], "true_t": p_true[3], "true_loge": p_true[4],
                "reco_x": vertex_time[0], "reco_y": vertex_time[1],
                "reco_z": vertex_time[2], "reco_t": vertex_time[3],
                "reco_theta": direction_params[5], "reco_phi": direction_params[6],
                "reco_loge": loge_gb,
                "initial_theta": theta0, "initial_phi": phi0,
                "dx": dx, "dy": dy, "dz": dz, "dt": dt,
                "angle_deg": angle, "dlogE": dloge,
                "vertex_ok": vertex_ok, "vertex_message": vertex_msg,
                "vertex_sampling_method": sampling_method,
                "vertex_selected_du": selected_du,
                "vertex_nll": vertex_nll, "time_s": elapsed,
            })

            gc.collect()

            # Convert angles to degrees for display
            gt_th_deg, gt_ph_deg = np.degrees(p_true[5]), np.degrees(p_true[6])
            reco_th_deg, reco_ph_deg = np.degrees(direction_params[5]), np.degrees(direction_params[6])

            # Format ground truth string
            gt_str = (f"{p_true[0]:6.2f}, {p_true[1]:6.2f}, {p_true[2]:6.2f}, "
                      f"{p_true[3]:6.2f}, {p_true[4]:5.2f}, {gt_th_deg:6.2f}, {gt_ph_deg:6.2f}")
            # Format reconstructed string
            reco_str = (f"{vertex_time[0]:6.2f}, {vertex_time[1]:6.2f}, {vertex_time[2]:6.2f}, "
                        f"{vertex_time[3]:6.2f}, {loge_gb:5.2f}, {reco_th_deg:6.2f}, {reco_ph_deg:6.2f}")

            print(f"{idx+1:>5} | {gt_str:^55} | {reco_str:^55} | {elapsed:>8.2f}")

    # Saving
    with open(f"{OUTPUT_PREFIX}_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    np.savez(f"{OUTPUT_PREFIX}_results.npz", **{k: np.asarray(v) for k, v in errors.items()})
    save_error_plots(errors)

    print("\nResidual Summaries")
    print(f"{'Variable':<15} | {'Mean':>8} | {'Median':>8} | {'P90 Abs':>8} | {'Max Abs':>8}")
    print("-" * 55)
    for key, label in [("dx", "dx [m]"), ("dy", "dy [m]"), ("dz", "dz [m]"),
                        ("dt", "dt"), ("angle", "angle [deg]"), ("dlogE", "dlogE")]:
        print(summarize(label, errors[key]))

    total = time.time() - t0_global
    print(f"\nSaved: {OUTPUT_PREFIX}_results.csv")
    print(f"Total time: {total:.1f}s  |  Average per event: {total/N_EVENTS:.1f}s")


if __name__ == "__main__":
    main()
