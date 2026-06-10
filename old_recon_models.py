"""
KM3NeT cascade reconstruction pipeline – file unico.
Esegue: caricamento eventi → ricostruzione vertice/tempo → stima energia (GB)
        → ricostruzione direzione (Fibonacci + fine grid) → salvataggio risultati.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import csv
import gc
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn.cluster import DBSCAN

# ── Configurazione run ───────────────────────────────────────────────────────

INPUT_FILES = [
    "events_100.npz",
    # "events_200.npz", ...
]

N_EVENTS      = 10
OUTPUT_PREFIX = "full_reco_gb"
RANDOM_SEED   = 12345

# ── Costanti fisiche ─────────────────────────────────────────────────────────

C_WATER        = 0.2222
SCALE_FACTOR   = 100.0
CALIBRATION_K  = 2.697575011e-05
K40_PE_PER_HIT = 1e-3

# ── Likelihood (binary occupancy) ────────────────────────────────────────────

OCCUPANCY_LIGHT_SCALE = 1.7682415367

# ── Ricostruzione vertice ─────────────────────────────────────────────────────

VERTEX_MAXITER         = 40
ENERGY_BOUNDS          = (3.5, 7.5)
N_VERTEX_RANDOM_STARTS = 10
MIN_VERTEX_DU          = 3
DU_CLUSTER_EPS_M       = 1.0
CAUSAL_PENALTY_WEIGHT  = 1.0
CAUSAL_PENALTY_TOL     = 20.0
CAUSAL_PENALTY_SCALE   = 10.0

# ── Ricostruzione direzione ───────────────────────────────────────────────────

COARSE_GRID_DEG    = 10.0
FIBONACCI_N_POINTS = 400
GB_MODEL_PATH      = "gb_energy_model.pkl"

# ── Variabili globali (modelli, scaler, geometria) ────────────────────────────

model_hits   = None
model_lambda = None
sh_mean = sh_std = None
sl_mean = sl_scale = None
P_POS   = None   # (N_pmt, 3)
P_DIR   = None   # (N_pmt, 3)


# ════════════════════════════════════════════════════════════════════════════
# I/O
# ════════════════════════════════════════════════════════════════════════════

def load_events_from_npz(npz_path):
    """Carica un archivio .npz e restituisce una lista di dizionari-evento."""
    global P_POS, P_DIR
    with np.load(npz_path) as data:
        P_POS = data["P_POS"]
        P_DIR = data["P_DIR"]
        p_signal_all = data["P_SIGNAL"].astype(np.float32)
        offsets  = data["hit_offsets"]
        n_events = len(data["p_true"])
        events   = []
        for i in range(n_events):
            s, e = offsets[i], offsets[i + 1]
            events.append({
                "p_true":        data["p_true"][i],
                "carica_totale": data["carica_totale"][i],
                "n_hits":        data["n_hits"][i],
                "H_POS":         data["hit_pos"][s:e],
                "H_DIR":         data["hit_dir"][s:e],
                "H_T":           data["hit_t"][s:e],
                "H_A":           data["hit_a"][s:e],
                "P_POS":         P_POS,
                "P_DIR":         P_DIR,
                "P_SIGNAL":      p_signal_all[i],
            })
    return events


def load_models():
    """Carica modelli Keras e scaler dal disco."""
    global model_hits, model_lambda, sh_mean, sh_std, sl_mean, sl_scale

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


def event_tensors(event):
    """Restituisce i tensori TF per un evento."""
    return (
        tf.constant(np.asarray(event["H_POS"],    dtype=np.float64).reshape(-1, 3), dtype=tf.float64),
        tf.constant(np.asarray(event["H_DIR"],    dtype=np.float64).reshape(-1, 3), dtype=tf.float64),
        tf.constant(np.asarray(event["H_T"],      dtype=np.float64).reshape(-1),    dtype=tf.float64),
        tf.constant(np.asarray(event["P_SIGNAL"], dtype=np.float64).reshape(-1),    dtype=tf.float64),
    )


# ════════════════════════════════════════════════════════════════════════════
# Likelihood
# ════════════════════════════════════════════════════════════════════════════

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(7,),    dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
    ]
)
def tf_nll_binary_occupancy(params, P_SIGNAL_OBS):
    """NLL Bernoulli basata su occupancy PMT."""
    s_pos = params[:3]
    log_e = params[4]
    theta = params[5]
    phi   = params[6]

    s_dir  = tf.stack([tf.sin(theta) * tf.cos(phi),
                       tf.sin(theta) * tf.sin(phi),
                       tf.cos(theta)])
    energy = tf.pow(tf.constant(10.0, dtype=tf.float64), log_e)

    v_p       = P_POS - s_pos
    dist_p    = tf.maximum(tf.norm(v_p, axis=1), 0.1)
    dist_safe = tf.minimum(dist_p, 1000.0)
    dir_p     = v_p / tf.expand_dims(dist_p, -1)
    cg_p      = tf.reduce_sum(s_dir * dir_p, axis=1)
    ca_p      = -tf.reduce_sum(dir_p * P_DIR, axis=1)
    logd_safe = tf.math.log1p(dist_safe)

    X_lam   = tf.stack([dist_safe, cg_p, ca_p, logd_safe], axis=1)
    X32_lam = tf.cast((tf.cast(X_lam, tf.float32) - sl_mean) / sl_scale, tf.float32)
    y_lam   = tf.cast(model_lambda(X32_lam, training=False), tf.float64)

    lam       = tf.maximum(
        tf.reshape(10.0 ** y_lam, [-1]) * energy * OCCUPANCY_LIGHT_SCALE, 0.0)
    att       = tf.exp(-(dist_p - dist_safe) / 60.0)
    mu_signal = lam * att

    mu    = mu_signal + K40_PE_PER_HIT
    p_hit = tf.maximum(-tf.math.expm1(-mu), 1e-12)
    hit   = tf.clip_by_value(P_SIGNAL_OBS, 0.0, 1.0)

    ll_term = hit * tf.math.log(p_hit) - (1.0 - hit) * mu
    return -tf.reduce_sum(ll_term)


def eval_nll(params_np, P_SIGNAL_tf):
    """Valuta la NLL per un vettore di parametri numpy."""
    return tf_nll_binary_occupancy(
        tf.constant(params_np, dtype=tf.float64),
        P_SIGNAL_tf,
    )


# ════════════════════════════════════════════════════════════════════════════
# Ricostruzione vertice e tempo
# ════════════════════════════════════════════════════════════════════════════

def build_du_lookup():
    labels = DBSCAN(eps=DU_CLUSTER_EPS_M, min_samples=10).fit_predict(P_POS[:, :2])
    if np.any(labels < 0):
        raise RuntimeError("DBSCAN ha lasciato PMT senza DU: aumenta DU_CLUSTER_EPS_M")
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
    hit_pos   = np.asarray(event["H_POS"], dtype=np.float64).reshape(-1, 3)
    hit_t     = np.asarray(event["H_T"],   dtype=np.float64).reshape(-1)
    first_idx = int(np.argmin(hit_t))
    first_pos = hit_pos[first_idx]
    centroid  = np.mean(hit_pos, axis=0)
    theta0, phi0 = angles_from_unit(centroid - first_pos)
    t0 = float(hit_t[first_idx]) - 50.0
    return first_pos, t0, theta0, phi0


def select_early_hits_on_multiple_dus(event, pmt_tree, du_labels):
    hit_pos    = np.asarray(event["H_POS"], dtype=np.float64).reshape(-1, 3)
    hit_t      = np.asarray(event["H_T"],   dtype=np.float64).reshape(-1)
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
    distances = np.linalg.norm(np.asarray(sel_pos) - np.asarray(pos)[None, :], axis=1)
    return float(np.median(np.asarray(sel_t) - distances / C_WATER))


def sample_farthest_points_in_hull(sel_pos, rng, n_samples, n_candidates=2000):
    sel_pos = np.asarray(sel_pos, dtype=np.float64)
    p_min, p_max = np.min(sel_pos, axis=0), np.max(sel_pos, axis=0)
    try:
        delaunay = Delaunay(sel_pos)
        pool     = rng.uniform(p_min, p_max, size=(n_candidates * 10, 3))
        inside   = pool[delaunay.find_simplex(pool) >= 0]
        if len(inside) < n_samples:
            return rng.uniform(p_min, p_max, size=(n_samples, 3)), "bounding_box"
    except QhullError:
        return rng.uniform(p_min, p_max, size=(n_samples, 3)), "bounding_box"
    centroid  = inside.mean(axis=0)
    first_idx = int(np.argmax(np.linalg.norm(inside - centroid, axis=1)))
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
    starts = []
    for pos in random_positions:
        starts.append(np.array([pos[0], pos[1], pos[2],
                                 initial_t0_from_position(pos, sel_pos, sel_t)],
                                dtype=np.float64))
    return starts, sampling_method, int(len(set(map(int, sel_du)))), sel_pos, sel_t


@tf.function
def vertex_time_nll_tf(x, fixed_theta_phi_loge, H_POS, H_DIR, H_T,
                        early_pos, early_t, causal_weight):
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
    time_res_h  = H_T - st - (dist_h / C_WATER)
    logd_h      = tf.math.log1p(dist_h)

    X_hits   = tf.stack([dist_h, cos_gamma_h, cos_alpha_h, time_res_h, logd_h], axis=1)
    X32_hits = tf.cast((tf.cast(X_hits, tf.float32) - sh_mean) / sh_std, tf.float32)
    y_hits   = tf.cast(model_hits({"input_1": X32_hits}, training=False), tf.float64)
    mu_hit   = tf.maximum(
        tf.reshape((tf.math.expm1(y_hits) / SCALE_FACTOR) * energy * CALIBRATION_K, [-1]), 0.0)
    nll = -tf.reduce_sum(tf.math.log(mu_hit + K40_PE_PER_HIT))

    early_dist        = tf.maximum(tf.norm(early_pos - s_pos, axis=1), 0.1)
    latest_allowed_t0 = early_t - early_dist / C_WATER
    violation         = st - latest_allowed_t0 - CAUSAL_PENALTY_TOL
    scaled_violation  = tf.nn.relu(violation / CAUSAL_PENALTY_SCALE)
    causal_penalty    = causal_weight * tf.reduce_mean(tf.square(scaled_violation))
    return nll + causal_penalty


def reconstruct_vertex_time(event, h_pos, h_dir, h_t, loge_seed, pmt_tree, du_labels, rng):
    _, _, theta0, phi0 = initial_vertex_and_direction(event)
    fixed = tf.constant([theta0, phi0, loge_seed], dtype=tf.float64)
    p_min = np.min(P_POS, axis=0) - 100.0
    p_max = np.max(P_POS, axis=0) + 100.0
    min_hit_t = float(np.min(np.asarray(event["H_T"], dtype=np.float64)))
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

    def scipy_fun(x_np):
        value, grad = value_and_grad(tf.constant(x_np, dtype=tf.float64))
        return float(value.numpy()), np.asarray(grad.numpy(), dtype=np.float64)

    best = None
    for start in starts:
        start  = np.array([np.clip(start[i], bounds[i][0], bounds[i][1]) for i in range(4)])
        result = minimize(fun=lambda x: scipy_fun(x), x0=start, method="L-BFGS-B", jac=True,
                          bounds=bounds,
                          options={"maxiter": VERTEX_MAXITER, "ftol": 1e-8, "gtol": 1e-5})
        if best is None or result.fun < best.fun:
            best = result

    return (best.x.astype(np.float64), theta0, phi0,
            bool(best.success), str(best.message),
            sampling_method, selected_du, float(best.fun))


# ════════════════════════════════════════════════════════════════════════════
# Stima energia (Gradient Boosting)
# ════════════════════════════════════════════════════════════════════════════

def gb_energy_guess(gb_model, vertex_time, n_hits):
    log_nhits = np.log10(max(float(n_hits), 1.0))
    reco_z    = float(vertex_time[2])
    r_vertex  = float(np.sqrt(vertex_time[0] ** 2 + vertex_time[1] ** 2))
    loge      = gb_model.predict([[log_nhits, reco_z, r_vertex]])[0]
    return float(np.clip(loge, *ENERGY_BOUNDS))


# ════════════════════════════════════════════════════════════════════════════
# Ricostruzione direzione (Fibonacci + fine grid)
# ════════════════════════════════════════════════════════════════════════════

def generate_fibonacci_sphere(n_samples):
    """Distribuisce n_samples punti uniformemente sulla sfera; restituisce (theta, phi)."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n_samples):
        y       = 1 - (i / float(n_samples - 1)) * 2
        radius  = np.sqrt(max(1 - y * y, 0.0))
        theta_g = 2 * np.pi * i / golden_ratio
        theta   = np.arccos(y)
        phi     = np.arctan2(np.sin(theta_g) * radius, np.cos(theta_g) * radius)
        points.append((theta, phi))
    return points


def run_angle_search(params_init, P_SIGNAL_tf):
    """
    Ricerca angolare in due passaggi:
      1) Coarse: sfera di Fibonacci (FIBONACCI_N_POINTS punti)
      2) Fine:   griglia 10x10 attorno al miglior punto coarse
    Posizione, tempo ed energia sono tenuti fissi al valore di params_init.
    Restituisce un dict con coarse_params, fine_params, space_angle_deg.
    """
    p_eval = np.array(params_init, dtype=np.float64, copy=True)

    # ── Fase 1: Fibonacci coarse ──────────────────────────────────────────
    best_coarse_nll   = float("inf")
    best_coarse_theta = p_eval[5]
    best_coarse_phi   = p_eval[6]

    for theta, phi in generate_fibonacci_sphere(FIBONACCI_N_POINTS):
        p_eval[5] = theta
        p_eval[6] = phi
        nll_val = float(eval_nll(p_eval, P_SIGNAL_tf).numpy())
        if np.isfinite(nll_val) and nll_val < best_coarse_nll:
            best_coarse_nll   = nll_val
            best_coarse_theta = theta
            best_coarse_phi   = phi

    # ── Fase 2: fine grid 10x10 ───────────────────────────────────────────
    half_cell       = np.radians(COARSE_GRID_DEG / 2.0)
    theta_fine_grid = np.linspace(max(0.0, best_coarse_theta - half_cell),
                                   min(np.pi, best_coarse_theta + half_cell), 10)
    phi_fine_grid   = np.linspace(best_coarse_phi - half_cell,
                                   best_coarse_phi + half_cell, 10)

    best_fine_nll   = float("inf")
    best_fine_theta = best_coarse_theta
    best_fine_phi   = best_coarse_phi

    for theta in theta_fine_grid:
        for phi in phi_fine_grid:
            p_eval[5] = theta
            p_eval[6] = phi
            nll_val = float(eval_nll(p_eval, P_SIGNAL_tf).numpy())
            if np.isfinite(nll_val) and nll_val < best_fine_nll:
                best_fine_nll   = nll_val
                best_fine_theta = theta
                best_fine_phi   = phi

    p_coarse    = np.array(params_init, dtype=np.float64, copy=True)
    p_coarse[5] = best_coarse_theta
    p_coarse[6] = best_coarse_phi

    p_fine    = np.array(params_init, dtype=np.float64, copy=True)
    p_fine[5] = best_fine_theta
    p_fine[6] = best_fine_phi

    dir_true  = unit_from_angles(params_init[5], params_init[6])
    dir_pred  = unit_from_angles(best_fine_theta, best_fine_phi)
    cos_alpha = np.clip(np.dot(dir_true, dir_pred), -1.0, 1.0)

    return {
        "coarse_params":   p_coarse,
        "fine_params":     p_fine,
        "space_angle_deg": float(np.degrees(np.arccos(cos_alpha))),
    }


# ════════════════════════════════════════════════════════════════════════════
# Utility
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
    return (f"{name}: mean={np.mean(values):.3f}, median={np.median(values):.3f}, "
            f"p90_abs={np.percentile(np.abs(values), 90):.3f}, "
            f"max_abs={np.max(np.abs(values)):.3f}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    load_models()

    with open(GB_MODEL_PATH, "rb") as f:
        gb_model = pickle.load(f)
    print(f"Modello GB caricato da {GB_MODEL_PATH}")

    rng    = np.random.default_rng(RANDOM_SEED)
    rows   = []
    errors = {"dx": [], "dy": [], "dz": [], "dt": [], "angle": [], "dlogE": []}
    t0_global = time.time()

    for input_file in INPUT_FILES:
        print(f"\n{'='*60}\nProcesso: {input_file}\n{'='*60}")

        events = load_events_from_npz(input_file)[:N_EVENTS]
        pmt_tree, du_labels = build_du_lookup()
        seed_loge = float(np.median([ev["p_true"][4] for ev in events]))
        print(f"  {len(events)} eventi  |  seed logE={seed_loge:.3f}")
        print("event,dx,dy,dz,dt,angle_deg,dlogE,vertex_ok,selected_du,sampling,time_s")

        for idx, event in enumerate(events):
            event_t0 = time.time()
            p_true   = np.asarray(event["p_true"], dtype=np.float64)
            n_hits   = int(event["n_hits"])
            h_pos, h_dir, h_t, p_signal = event_tensors(event)

            # 1. Vertice e tempo
            (vertex_time, theta0, phi0,
             vertex_ok, vertex_msg,
             sampling_method, selected_du,
             vertex_nll) = reconstruct_vertex_time(
                event, h_pos, h_dir, h_t, seed_loge, pmt_tree, du_labels, rng)

            # 2. Energia (GB)
            loge_gb = gb_energy_guess(gb_model, vertex_time, n_hits)

            # 3. Direzione
            params_for_angle     = np.array(p_true, dtype=np.float64, copy=True)
            params_for_angle[:3] = vertex_time[:3]
            params_for_angle[3]  = vertex_time[3]
            params_for_angle[4]  = loge_gb
            direction_result     = run_angle_search(params_for_angle, p_signal)
            direction_params     = np.asarray(direction_result["fine_params"], dtype=np.float64)

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
            print(f"{idx+1},{dx:.3f},{dy:.3f},{dz:.3f},{dt:.3f},"
                  f"{angle:.3f},{dloge:.3f},{vertex_ok},"
                  f"{selected_du},{sampling_method},{elapsed:.2f}")

    # Salvataggio
    with open(f"{OUTPUT_PREFIX}_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    np.savez(f"{OUTPUT_PREFIX}_results.npz", **{k: np.asarray(v) for k, v in errors.items()})
    #save_error_plots(errors)

    print("\nSummaries")
    for key, label in [("dx", "dx [m]"), ("dy", "dy [m]"), ("dz", "dz [m]"),
                        ("dt", "dt"), ("angle", "angle [deg]"), ("dlogE", "dlogE")]:
        print(summarize(label, errors[key]))

    total = time.time() - t0_global
    print(f"\nSalvato: {OUTPUT_PREFIX}_results.csv")
    print(f"Tempo totale: {total:.1f}s  |  Medio per evento: {total/N_EVENTS:.1f}s")


if __name__ == "__main__":
    main()
