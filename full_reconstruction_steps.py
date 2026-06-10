import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn.cluster import DBSCAN

import main

# ── File da processare in sequenza ───────────────────────────────────────────

INPUT_FILES = [
    "events_100.npz",
    #"calibratioN_EVENTS_200.npz",
    #"calibratioN_EVENTS_300_1.npz",
    #"calibratioN_EVENTS_300_2.npz",
    #"calibratioN_EVENTS_300_3.npz",
]

N_EVENTS = 100


VERTEX_MAXITER         = 80
ENERGY_BOUNDS          = (3.5, 7.5)
OUTPUT_PREFIX          = "full_reco_gb"
N_VERTEX_RANDOM_STARTS = 10
MIN_VERTEX_DU          = 3
DU_CLUSTER_EPS_M       = 1.0
RANDOM_SEED            = 12345
CAUSAL_PENALTY_WEIGHT  = 1.0
CAUSAL_PENALTY_TOL     = 20.0
CAUSAL_PENALTY_SCALE   = 10.0
GB_MODEL_PATH          = "gb_energy_model.pkl"


def load_models():
    main.model_hits = tf.keras.models.load_model("model.h5", compile=False)
    with open("scalers.pkl", "rb") as f:
        scalers_hits = pickle.load(f)

    main.model_lambda = tf.keras.models.load_model("model_lambda.h5", compile=False)
    with open("scalers_lambda.pkl", "rb") as f:
        scalers_lambda = pickle.load(f)

    main.sh_mean = tf.constant(
        [scalers_hits["dist"]["mean"], 0.0, 0.0,
         scalers_hits["time"]["mean"], scalers_hits["log_dist"]["mean"]],
        dtype=tf.float32,
    )
    main.sh_std = tf.constant(
        [scalers_hits["dist"]["std"], 1.0, 1.0,
         scalers_hits["time"]["std"], scalers_hits["log_dist"]["std"]],
        dtype=tf.float32,
    )
    main.sl_mean  = tf.constant(scalers_lambda.mean_,  dtype=tf.float32)
    main.sl_scale = tf.constant(scalers_lambda.scale_, dtype=tf.float32)


def event_tensors(event):
    return (
        tf.constant(np.asarray(event["H_POS"], dtype=np.float64).reshape(-1, 3), dtype=tf.float64),
        tf.constant(np.asarray(event["H_DIR"], dtype=np.float64).reshape(-1, 3), dtype=tf.float64),
        tf.constant(np.asarray(event["H_T"],   dtype=np.float64).reshape(-1),    dtype=tf.float64),
        tf.constant(np.asarray(event["P_SIGNAL"], dtype=np.float64).reshape(-1), dtype=tf.float64),
    )


def build_du_lookup():
    labels = DBSCAN(eps=DU_CLUSTER_EPS_M, min_samples=10).fit_predict(main.P_POS[:, :2])
    if np.any(labels < 0):
        raise RuntimeError("DBSCAN ha lasciato PMT senza DU: aumenta DU_CLUSTER_EPS_M")
    return cKDTree(main.P_POS), labels.astype(np.int32)


def unit_from_angles(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)])


def angles_from_unit(vec):
    norm = np.linalg.norm(vec)
    if norm == 0.0 or not np.isfinite(norm):
        return np.pi / 2.0, 0.0
    vec = vec / norm
    theta = np.arccos(np.clip(vec[2], -1.0, 1.0))
    phi   = np.arctan2(vec[1], vec[0])
    return float(theta), float(phi)


def angular_error_deg(true_theta, true_phi, reco_theta, reco_phi):
    true_dir = unit_from_angles(true_theta, true_phi)
    reco_dir = unit_from_angles(reco_theta, reco_phi)
    return float(np.degrees(np.arccos(np.clip(np.dot(true_dir, reco_dir), -1.0, 1.0))))


def initial_vertex_and_direction(event):
    hit_pos  = np.asarray(event["H_POS"], dtype=np.float64).reshape(-1, 3)
    hit_t    = np.asarray(event["H_T"],   dtype=np.float64).reshape(-1)
    first_idx = int(np.argmin(hit_t))
    first_pos = hit_pos[first_idx]
    centroid  = np.mean(hit_pos, axis=0)
    theta0, phi0 = angles_from_unit(centroid - first_pos)
    t0 = float(hit_t[first_idx]) - 50.0
    return first_pos, t0, theta0, phi0


def select_early_hits_on_multiple_dus(event, pmt_tree, du_labels):
    hit_pos = np.asarray(event["H_POS"], dtype=np.float64).reshape(-1, 3)
    hit_t   = np.asarray(event["H_T"],   dtype=np.float64).reshape(-1)
    _, pmt_idx = pmt_tree.query(hit_pos, k=1)
    hit_du  = du_labels[pmt_idx]
    order   = np.argsort(hit_t)
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
    return float(np.median(np.asarray(sel_t) - distances / main.C_WATER))


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
    random_positions, sampling_method = sample_farthest_points_in_hull(sel_pos, rng, N_VERTEX_RANDOM_STARTS)
    starts = []
    for pos in random_positions:
        starts.append(np.array([pos[0], pos[1], pos[2],
                                 initial_t0_from_position(pos, sel_pos, sel_t)], dtype=np.float64))
    return starts, sampling_method, int(len(set(map(int, sel_du)))), sel_pos, sel_t


@tf.function
def vertex_time_nll_tf(x, fixed_theta_phi_loge, H_POS, H_DIR, H_T, early_pos, early_t, causal_weight):
    s_pos = x[:3]; st = x[3]
    theta, phi, log_e = fixed_theta_phi_loge[0], fixed_theta_phi_loge[1], fixed_theta_phi_loge[2]
    s_dir  = tf.stack([tf.sin(theta)*tf.cos(phi), tf.sin(theta)*tf.sin(phi), tf.cos(theta)])
    energy = tf.pow(tf.constant(10.0, dtype=tf.float64), log_e)
    v_h         = H_POS - s_pos
    dist_h      = tf.maximum(tf.norm(v_h, axis=1), 0.1)
    dir_h       = v_h / tf.expand_dims(dist_h, -1)
    cos_gamma_h = tf.reduce_sum(s_dir * dir_h, axis=1)
    cos_alpha_h = -tf.reduce_sum(dir_h * H_DIR, axis=1)
    time_res_h  = H_T - st - (dist_h / main.C_WATER)
    logd_h      = tf.math.log1p(dist_h)
    X_hits   = tf.stack([dist_h, cos_gamma_h, cos_alpha_h, time_res_h, logd_h], axis=1)
    X32_hits = tf.cast((tf.cast(X_hits, tf.float32) - main.sh_mean) / main.sh_std, tf.float32)
    y_hits   = tf.cast(main.model_hits({"input_1": X32_hits}, training=False), tf.float64)
    mu_hit   = tf.maximum(tf.reshape((tf.math.expm1(y_hits) / main.SCALE_FACTOR) * energy * main.CALIBRATION_K, [-1]), 0.0)
    nll      = -tf.reduce_sum(tf.math.log(mu_hit + main.K40_PE_PER_HIT))
    early_dist        = tf.maximum(tf.norm(early_pos - s_pos, axis=1), 0.1)
    latest_allowed_t0 = early_t - early_dist / main.C_WATER
    violation         = st - latest_allowed_t0 - CAUSAL_PENALTY_TOL
    scaled_violation  = tf.nn.relu(violation / CAUSAL_PENALTY_SCALE)
    causal_penalty    = causal_weight * tf.reduce_mean(tf.square(scaled_violation))
    return nll + causal_penalty


def reconstruct_vertex_time(event, h_pos, h_dir, h_t, loge_seed, pmt_tree, du_labels, rng):
    _, _, theta0, phi0 = initial_vertex_and_direction(event)
    fixed  = tf.constant([theta0, phi0, loge_seed], dtype=tf.float64)
    p_min  = np.min(main.P_POS, axis=0) - 100.0
    p_max  = np.max(main.P_POS, axis=0) + 100.0
    min_hit_t = float(np.min(np.asarray(event["H_T"], dtype=np.float64)))
    bounds = [(float(p_min[0]), float(p_max[0])),
              (float(p_min[1]), float(p_max[1])),
              (float(p_min[2]), float(p_max[2])),
              (-200.0, min_hit_t + 20.0)]

    starts, sampling_method, selected_du, sel_pos, sel_t = make_random_vertex_starts(
        event, pmt_tree, du_labels, rng)
    early_pos_tf    = tf.constant(sel_pos, dtype=tf.float64)
    early_t_tf      = tf.constant(sel_t,   dtype=tf.float64)
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
                          bounds=bounds, options={"maxiter": VERTEX_MAXITER, "ftol": 1e-8, "gtol": 1e-5})
        if best is None or result.fun < best.fun:
            best = result

    return (best.x.astype(np.float64), theta0, phi0,
            bool(best.success), str(best.message),
            sampling_method, selected_du, float(best.fun))


def gb_energy_guess(gb_model, vertex_time, n_hits):
    """Stima l'energia con il Gradient Boosting usando posizione ricostruita e n_hits."""
    log_nhits = np.log10(max(float(n_hits), 1.0))
    reco_z    = float(vertex_time[2])
    r_vertex  = float(np.sqrt(vertex_time[0]**2 + vertex_time[1]**2))
    loge      = gb_model.predict([[log_nhits, reco_z, r_vertex]])[0]
    return float(np.clip(loge, *ENERGY_BOUNDS))


def direction_reconstruction(event, vertex_time, loge_guess, h_pos, h_dir, h_t, p_signal):
    params = np.asarray(event["p_true"], dtype=np.float64).copy()
    params[:3] = vertex_time[:3]
    params[3]  = vertex_time[3]
    params[4]  = loge_guess
    return main.run_angle_search(params, h_pos, h_dir, h_t, p_signal, 0.0, 1.0)


def save_error_plots(errors):
    plot_specs = [
        ("dx",    "X residual [m]",    "X residual distribution"),
        ("dy",    "Y residual [m]",    "Y residual distribution"),
        ("dz",    "Z residual [m]",    "Z residual distribution"),
        ("dt",    "T residual",        "Time residual distribution"),
        ("angle", "Angular error [deg]", "Direction angular error distribution"),
        ("dlogE", "log10(E) residual", "log10(E) residual distribution"),
    ]
    for key, xlabel, title in plot_specs:
        values = np.asarray(errors[key], dtype=np.float64)
        plt.figure(figsize=(7.5, 5.0))
        plt.hist(values, bins=30, alpha=0.75, color="#4c78a8")
        plt.axvline(np.median(values), color="#f58518", linestyle="--", linewidth=2.0, label="median")
        plt.xlabel(xlabel); plt.ylabel("Events"); plt.title(title)
        plt.grid(axis="y", alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(f"{OUTPUT_PREFIX}_{key}_distribution.png", dpi=180)
        plt.close()


def summarize(name, values):
    values = np.asarray(values, dtype=np.float64)
    return (f"{name}: mean={np.mean(values):.3f}, median={np.median(values):.3f}, "
            f"p90_abs={np.percentile(np.abs(values), 90):.3f}, max_abs={np.max(np.abs(values)):.3f}")


def main_run():
    load_models()
    main.LIKELIHOOD_MODE             = "binary_occupancy"
    main.OCCUPANCY_USE_SIGMOID_WEIGHTS = False
    main.OCCUPANCY_LIGHT_SCALE       = 0.655493
    rng = np.random.default_rng(RANDOM_SEED)

    # Carica il modello GB
    with open(GB_MODEL_PATH, "rb") as f:
        gb_model = pickle.load(f)
    print(f"Modello GB caricato da {GB_MODEL_PATH}")

    rows  = []
    errors = {"dx": [], "dy": [], "dz": [], "dt": [], "angle": [], "dlogE": []}
    t0_global = time.time()

    for input_file in INPUT_FILES:
        print(f"\n{'='*60}")
        print(f"Processo: {input_file}")
        print(f"{'='*60}")

        # Carica eventi e aggiorna geometria globale
        main.events_ = main.load_events_from_npz(input_file)
        main.events_ = main.events_[:N_EVENTS]

        # Ricostruisce il lookup DU sulla geometria del file corrente
        pmt_tree, du_labels = build_du_lookup()

        # Seed di energia: mediana del GT degli eventi correnti
        seed_loge = float(np.median([ev["p_true"][4] for ev in main.events_]))
        print(f"  {N_EVENTS} eventi  |  seed logE={seed_loge:.3f}")
        print("event,dx,dy,dz,dt,angle_deg,dlogE,vertex_ok,selected_du,sampling,time_s")

        for idx, event in enumerate(main.events_):
            event_t0 = time.time()
            p_true   = np.asarray(event["p_true"], dtype=np.float64)
            n_hits   = int(event["n_hits"])
            h_pos, h_dir, h_t, p_signal = event_tensors(event)

            # ── 1. Ricostruzione posizione e tempo ────────────────────────────
            (vertex_time, theta0, phi0,
             vertex_ok, vertex_msg,
             sampling_method, selected_du,
             vertex_nll) = reconstruct_vertex_time(
                event, h_pos, h_dir, h_t, seed_loge, pmt_tree, du_labels, rng)

            # ── 2. Stima energia con Gradient Boosting ────────────────────────
            loge_gb = gb_energy_guess(gb_model, vertex_time, n_hits)

            # ── 3. Ricostruzione direzione con Fibonacci ──────────────────────
            direction_result = direction_reconstruction(
                event, vertex_time, loge_gb, h_pos, h_dir, h_t, p_signal)
            direction_params = np.asarray(direction_result["fine_params"], dtype=np.float64)

            dx      = float(vertex_time[0] - p_true[0])
            dy      = float(vertex_time[1] - p_true[1])
            dz      = float(vertex_time[2] - p_true[2])
            dt      = float(vertex_time[3] - p_true[3])
            angle   = angular_error_deg(p_true[5], p_true[6], direction_params[5], direction_params[6])
            dloge   = float(loge_gb - p_true[4])
            elapsed = time.time() - event_t0

            for key, val in zip(["dx","dy","dz","dt","angle","dlogE"],
                                 [dx, dy, dz, dt, angle, dloge]):
                errors[key].append(val)

            row = {
                "file":                   input_file,
                "event":                  idx + 1,
                "true_x":                 p_true[0], "true_y": p_true[1],
                "true_z":                 p_true[2], "true_t": p_true[3],
                "true_loge":              p_true[4],
                "reco_x":                 vertex_time[0], "reco_y": vertex_time[1],
                "reco_z":                 vertex_time[2], "reco_t": vertex_time[3],
                "reco_theta":             direction_params[5], "reco_phi": direction_params[6],
                "reco_loge":              loge_gb,
                "initial_theta":          theta0, "initial_phi": phi0,
                "dx": dx, "dy": dy, "dz": dz, "dt": dt,
                "angle_deg":              angle, "dlogE": dloge,
                "vertex_ok":              vertex_ok,
                "vertex_message":         vertex_msg,
                "vertex_sampling_method": sampling_method,
                "vertex_selected_du":     selected_du,
                "vertex_nll":             vertex_nll,
                "time_s":                 elapsed,
            }
            rows.append(row)

            del main.events_
            main.events_ = []
            import gc
            gc.collect()

            print(f"{idx+1},{dx:.3f},{dy:.3f},{dz:.3f},{dt:.3f},"
                  f"{angle:.3f},{dloge:.3f},{vertex_ok},"
                  f"{selected_du},{sampling_method},{elapsed:.2f}")

    # ── Salvataggio risultati aggregati ───────────────────────────────────────
    with open(f"{OUTPUT_PREFIX}_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    np.savez(f"{OUTPUT_PREFIX}_results.npz",
             **{k: np.asarray(v) for k, v in errors.items()})
    save_error_plots(errors)

    print("\nSummaries")
    for key, label in [("dx","dx [m]"),("dy","dy [m]"),("dz","dz [m]"),
                        ("dt","dt"),("angle","angle [deg]"),("dlogE","dlogE")]:
        print(summarize(label, errors[key]))
    print(f"\nSaved CSV:   {OUTPUT_PREFIX}_results.csv")
    print(f"Total time:  {time.time()-t0_global:.1f}s")
    print(f"Average time per event: {(time.time()-t0_global)/N_EVENTS:.1f}s")


if __name__ == "__main__":
    main_run()
