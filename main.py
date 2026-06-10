import os
import time
import sys
import ctypes
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def load_events_from_npz(npz_path):
    """
    Carica un archivio .npz e lo ricostruisce nella struttura originale
    a lista di dizionari per garantire compatibilità con le funzioni esistenti.
    """
    with np.load(npz_path) as data:
        events = []
        n_events = len(data["p_true"])
        offsets = data["hit_offsets"]

        # Estraiamo la geometria globale
        global P_POS, P_DIR
        P_POS = data["P_POS"]
        P_DIR = data["P_DIR"]

        p_signal_all = data["P_SIGNAL"].astype(np.float32)

        for i in range(n_events):
            start, end = offsets[i], offsets[i+1]
            #p_signal_all = data["P_SIGNAL"].astype(np.float32)

            ev = {
                "p_true": data["p_true"][i],
                "carica_totale": data["carica_totale"][i],
                "n_hits": data["n_hits"][i],

                # Tagliamo la fetta esatta di hit appartenenti a questo evento
                "H_POS": data["hit_pos"][start:end],
                "H_DIR": data["hit_dir"][start:end],
                "H_T": data["hit_t"][start:end],
                "H_A": data["hit_a"][start:end],

                # Assegniamo i puntatori alla geometria globale (costo zero in RAM)
                "P_POS": P_POS,
                "P_DIR": P_DIR,
                "P_SIGNAL": p_signal_all[i]
            }
            events.append(ev)

    return events

# Esempio di utilizzo:
events_ = load_events_from_npz("events_100.npz")
print("done")


# ── Costanti Fisiche e di Calibrazione ──────────────────────────────────────
C_WATER          = 0.2222
SCALE_FACTOR     = 100.0
CALIBRATION_K    = 2.697575011e-05
K40_PE_PER_HIT = 1e-3

# Parametri sigmoide per la rete Lambda (Energia, base 10)
LAMBDA_CUTOFF_THRESHOLD = 2.0
LAMBDA_CUTOFF_DAMPING   = 7.5

# Parametri sigmoide per la rete Geometrica (Hits, base e)
soglia_vals      = [10]
k_ripidita_vals  = [10000]

# I PMT del dataset sono saturati: il dato osservabile e' binario
# (hit / non-hit), non un conteggio di fotoni. La likelihood principale
# usa quindi la probabilita' Bernoulli di accensione del PMT.
LIKELIHOOD_MODE = "binary_occupancy"
OCCUPANCY_LIGHT_SCALE = 1.0
OCCUPANCY_EPS = 1e-12
OCCUPANCY_USE_SIGMOID_WEIGHTS = False

# Griglia angolare COARSE (10 gradi)
COARSE_GRID_DEG = 10.0
_n_theta_c = int(round(180.0 / COARSE_GRID_DEG)) + 1
_n_phi_c   = int(round(360.0 / COARSE_GRID_DEG)) + 1
THETA_GRID_COARSE = np.linspace(0.0, np.pi, _n_theta_c)
PHI_GRID_COARSE   = np.linspace(-np.pi, np.pi, _n_phi_c)



@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(7,), dtype=tf.float64),        # params
        tf.TensorSpec(shape=(None, 3), dtype=tf.float64),   # H_POS
        tf.TensorSpec(shape=(None, 3), dtype=tf.float64),   # H_DIR
        tf.TensorSpec(shape=(None,), dtype=tf.float64),     # H_T
        tf.TensorSpec(shape=(), dtype=tf.float64),          # geom_ct
        tf.TensorSpec(shape=(), dtype=tf.float64),          # geom_cd
        tf.TensorSpec(shape=(), dtype=tf.float64),          # lam_ct
        tf.TensorSpec(shape=(), dtype=tf.float64),          # lam_cd
    ]
)
def tf_nll_eval_legacy(params, H_POS, H_DIR, H_T, geom_ct, geom_cd, lam_ct, lam_cd):
    s_pos = params[:3]
    st    = params[3]
    log_e = params[4]
    theta = params[5]
    phi   = params[6]

    s_dir = tf.stack([
        tf.sin(theta) * tf.cos(phi),
        tf.sin(theta) * tf.sin(phi),
        tf.cos(theta),
    ])
    energy = tf.pow(tf.constant(10.0, dtype=tf.float64), log_e)

    # ── Termine Hits ──
    v_h         = H_POS - s_pos
    dist_h      = tf.maximum(tf.norm(v_h, axis=1), 0.1)
    dir_h       = v_h / tf.expand_dims(dist_h, -1)
    cos_gamma_h = tf.reduce_sum(s_dir * dir_h, axis=1)
    cos_alpha_h = -tf.reduce_sum(dir_h * H_DIR, axis=1)
    time_res_h  = H_T - st - (dist_h / C_WATER)
    logd_h      = tf.math.log1p(dist_h)

    X_hits = tf.stack([dist_h, cos_gamma_h, cos_alpha_h, time_res_h, logd_h], axis=1)
    X32_hits = tf.cast((tf.cast(X_hits, tf.float32) - sh_mean) / sh_std, tf.float32)
    y_hits = tf.cast(model_hits({"input_1": X32_hits}, training=False), tf.float64)

    sig_pe = tf.reshape((tf.math.expm1(y_hits) / SCALE_FACTOR) * energy * CALIBRATION_K, [-1])
    ll_term = tf.math.log(sig_pe + K40_PE_PER_HIT) # Usa K40 = 1e-5

    weights_geom = tf.math.sigmoid((geom_ct - sig_pe) / geom_cd)
    ll = tf.reduce_sum(weights_geom * ll_term)

    # ── Termine Lambda ──
    v_p       = P_POS - s_pos
    dist_p    = tf.maximum(tf.norm(v_p, axis=1), 0.1)
    dist_safe = tf.minimum(dist_p, 1000.0)
    dir_p     = v_p / tf.expand_dims(dist_p, -1)
    cg_p      = tf.reduce_sum(s_dir * dir_p, axis=1)
    ca_p      = -tf.reduce_sum(dir_p * P_DIR, axis=1)
    logd_safe = tf.math.log1p(dist_safe)

    X_lam = tf.stack([dist_safe, cg_p, ca_p, logd_safe], axis=1)
    X32_lam = tf.cast((tf.cast(X_lam, tf.float32) - sl_mean) / sl_scale, tf.float32)
    y_lam = tf.cast(model_lambda(X32_lam, training=False), tf.float64)

    lam = tf.maximum(
        tf.reshape((10.0 ** y_lam) / SCALE_FACTOR, [-1]) * energy * CALIBRATION_K,
        0.0,
    )
    att   = tf.exp(-(dist_p - dist_safe) / 60.0)
    lam_w = lam * att
    weights_lam = tf.math.sigmoid((lam_ct - lam_w) / lam_cd)

    # Questo è l'equivalente del tuo vecchio 'lam'
    lam_sum = tf.reduce_sum(weights_lam * lam_w)

    # --- RITORNO ALLA FORMULAZIONE ORIGINALE ---
    # Extended Maximum Likelihood: NLL = - (logL_hits - Expected_hits)
    return -(ll - lam_sum)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(7,), dtype=tf.float64),        # params
        tf.TensorSpec(shape=(None,), dtype=tf.float64),     # P_SIGNAL, binario
        tf.TensorSpec(shape=(), dtype=tf.float64),          # lam_ct
        tf.TensorSpec(shape=(), dtype=tf.float64),          # lam_cd
        tf.TensorSpec(shape=(), dtype=tf.float64),          # light_scale
        tf.TensorSpec(shape=(), dtype=tf.bool),             # use_sigmoid_weights
    ]
)
def tf_nll_eval_binary_occupancy(
    params,
    P_SIGNAL_OBS,
    weight_threshold,
    weight_damping,
    light_scale,
    use_sigmoid_weights,
):
    s_pos = params[:3]
    log_e = params[4]
    theta = params[5]
    phi   = params[6]

    s_dir = tf.stack([
        tf.sin(theta) * tf.cos(phi),
        tf.sin(theta) * tf.sin(phi),
        tf.cos(theta),
    ])
    energy = tf.pow(tf.constant(10.0, dtype=tf.float64), log_e)

    v_p       = P_POS - s_pos
    dist_p    = tf.maximum(tf.norm(v_p, axis=1), 0.1)
    dist_safe = tf.minimum(dist_p, 1000.0)
    dir_p     = v_p / tf.expand_dims(dist_p, -1)
    cg_p      = tf.reduce_sum(s_dir * dir_p, axis=1)
    ca_p      = -tf.reduce_sum(dir_p * P_DIR, axis=1)
    logd_safe = tf.math.log1p(dist_safe)

    X_lam = tf.stack([dist_safe, cg_p, ca_p, logd_safe], axis=1)
    X32_lam = tf.cast((tf.cast(X_lam, tf.float32) - sl_mean) / sl_scale, tf.float32)
    y_lam = tf.cast(model_lambda(X32_lam, training=False), tf.float64)

    # La rete lambda e' addestrata con target log10(expected_hits), quindi
    # 10**y_lam e' gia' sulla scala fisica del dataset di training.
    lam = tf.maximum(
        tf.reshape(10.0 ** y_lam, [-1]) * energy * CALIBRATION_K * light_scale,
        0.0,
    )
    att = tf.exp(-(dist_p - dist_safe) / 60.0)
    mu_signal = lam * att

    # Per un PMT saturato l'informazione e' P(hit), non il numero di fotoni.
    # P(hit) = 1 - exp(-(mu_signal + mu_background)).
    mu = mu_signal + K40_PE_PER_HIT
    p_hit = tf.maximum(-tf.math.expm1(-mu), OCCUPANCY_EPS)
    hit = tf.clip_by_value(P_SIGNAL_OBS, 0.0, 1.0)

    ll_term = hit * tf.math.log(p_hit) - (1.0 - hit) * mu
    safe_damping = tf.where(
        tf.abs(weight_damping) < 1e-12,
        tf.constant(1e-12, dtype=tf.float64),
        weight_damping,
    )
    weights = tf.cond(
        use_sigmoid_weights,
        lambda: tf.math.sigmoid((weight_threshold - mu_signal) / safe_damping),
        lambda: tf.ones_like(mu_signal),
    )

    log_likelihood = tf.reduce_sum(weights * ll_term)
    return -log_likelihood




def generate_fibonacci_sphere(n_samples):
    """Genera n_samples punti distribuiti uniformemente sulla sfera (restituisce theta, phi)."""
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_samples):
        # y va da 1 (Polo Nord) a -1 (Polo Sud)
        y = 1 - (i / float(n_samples - 1)) * 2
        radius = np.sqrt(1 - y * y)

        theta_golden = 2 * np.pi * i / golden_ratio

        x = np.cos(theta_golden) * radius
        z = np.sin(theta_golden) * radius

        # Convertiamo nelle nostre coordinate:
        # Theta: 0 (su) -> pi (giù). Coincide con arccos(y)
        # Phi: azimut -pi -> pi. Coincide con arctan2(z, x)
        theta = np.arccos(y)
        phi = np.arctan2(z, x)
        points.append((theta, phi))
    return points

def run_angle_search(p_true, H_POS, H_DIR, H_T, P_SIGNAL_OBS, geom_ct_val, geom_cd_val):
    """
    Scansione in due passaggi:
    1) Grid coarse 10 gradi sull'intera sfera
    2) Grid fine 10x10 all'interno della cella trovata al passo 1
    I parametri x, y, z, t, E restano fissi a quelli del ground truth.
    """
    # Costanti TF per i cutoff
    lam_ct_tf  = tf.constant(LAMBDA_CUTOFF_THRESHOLD, dtype=tf.float64)
    lam_cd_tf  = tf.constant(LAMBDA_CUTOFF_DAMPING, dtype=tf.float64)
    geom_ct_tf = tf.constant(geom_ct_val, dtype=tf.float64)
    geom_cd_tf = tf.constant(geom_cd_val, dtype=tf.float64)
    occupancy_light_scale_tf = tf.constant(OCCUPANCY_LIGHT_SCALE, dtype=tf.float64)
    occupancy_use_sigmoid_weights_tf = tf.constant(OCCUPANCY_USE_SIGMOID_WEIGHTS)

    p_eval = np.array(p_true, dtype=np.float64, copy=True)

    # ── FASE 1: Grid Search Coarse (Fibonacci Sphere) ──
    best_coarse_nll = float("inf")
    best_coarse_theta = p_eval[5]
    best_coarse_phi   = p_eval[6]

    # 400 punti sono più che sufficienti per avere una risoluzione di ~10°
    coarse_points = generate_fibonacci_sphere(400)

    for theta, phi in coarse_points:
        p_eval[5] = theta
        p_eval[6] = phi
        if LIKELIHOOD_MODE == "legacy_eml":
            nll = tf_nll_eval_legacy(
                tf.convert_to_tensor(p_eval, dtype=tf.float64),
                H_POS, H_DIR, H_T,
                geom_ct_tf, geom_cd_tf, lam_ct_tf, lam_cd_tf
            )
        elif LIKELIHOOD_MODE == "binary_occupancy":
            nll = tf_nll_eval_binary_occupancy(
                tf.convert_to_tensor(p_eval, dtype=tf.float64),
                P_SIGNAL_OBS,
                geom_ct_tf,
                geom_cd_tf,
                occupancy_light_scale_tf,
                occupancy_use_sigmoid_weights_tf,
            )
        else:
            raise ValueError(f"Likelihood mode sconosciuta: {LIKELIHOOD_MODE}")
        nll_val = float(nll.numpy())
        if np.isfinite(nll_val) and nll_val < best_coarse_nll:
            best_coarse_nll   = nll_val
            best_coarse_theta = theta
            best_coarse_phi   = phi

    p_coarse = np.array(p_true, dtype=np.float64, copy=True)
    p_coarse[5] = best_coarse_theta
    p_coarse[6] = best_coarse_phi

    # ── FASE 2: Grid Search Fine (Subgrid 10x10) ──
    # Una cella da 10 gradi centrata sul best coarse significa +/- 5 gradi
    half_cell = np.radians(COARSE_GRID_DEG / 2.0)

    theta_fine_grid = np.linspace(
        max(0.0, best_coarse_theta - half_cell),
        min(np.pi, best_coarse_theta + half_cell),
        10
    )
    phi_fine_grid = np.linspace(
        best_coarse_phi - half_cell,
        best_coarse_phi + half_cell,
        10
    )

    best_fine_nll = float("inf")
    best_fine_theta = best_coarse_theta
    best_fine_phi   = best_coarse_phi

    for theta in theta_fine_grid:
        for phi in phi_fine_grid:
            p_eval[5] = theta
            p_eval[6] = phi
            if LIKELIHOOD_MODE == "legacy_eml":
                nll = tf_nll_eval_legacy(
                    tf.convert_to_tensor(p_eval, dtype=tf.float64),
                    H_POS, H_DIR, H_T,
                    geom_ct_tf, geom_cd_tf, lam_ct_tf, lam_cd_tf
                )
            elif LIKELIHOOD_MODE == "binary_occupancy":
                nll = tf_nll_eval_binary_occupancy(
                    tf.convert_to_tensor(p_eval, dtype=tf.float64),
                    P_SIGNAL_OBS,
                    geom_ct_tf,
                    geom_cd_tf,
                    occupancy_light_scale_tf,
                    occupancy_use_sigmoid_weights_tf,
                )
            else:
                raise ValueError(f"Likelihood mode sconosciuta: {LIKELIHOOD_MODE}")
            nll_val = float(nll.numpy())
            if np.isfinite(nll_val) and nll_val < best_fine_nll:
                best_fine_nll   = nll_val
                best_fine_theta = theta
                best_fine_phi   = phi

    p_fine = np.array(p_true, dtype=np.float64, copy=True)
    p_fine[5] = best_fine_theta
    p_fine[6] = best_fine_phi

    # ── Calcolo dell'angolo spaziale 3D ──
    dir_true = np.array([
        np.sin(p_true[5]) * np.cos(p_true[6]),
        np.sin(p_true[5]) * np.sin(p_true[6]),
        np.cos(p_true[5])
    ])

    dir_pred = np.array([
        np.sin(best_fine_theta) * np.cos(best_fine_phi),
        np.sin(best_fine_theta) * np.sin(best_fine_phi),
        np.cos(best_fine_theta)
    ])

    cos_alpha = np.clip(np.dot(dir_true, dir_pred), -1.0, 1.0)
    space_angle_deg = np.degrees(np.arccos(cos_alpha))

    return {
        "coarse_params": p_coarse,
        "fine_params": p_fine,
        "space_angle_deg": space_angle_deg,
    }

def _fmt(params):
    return (
        f"X={params[0]:.2f}, Y={params[1]:.2f}, Z={params[2]:.2f}, "
        f"T={params[3]:.2f}, Log10(E)={params[4]:.2f}, "
        f"Theta={params[5]:.4f}, Phi={params[6]:.4f}"
    )

def format_ground_truth(params):
    return _fmt(params)


def build_hit_indices(H_POS, tol=1e-3):
    if len(H_POS) == 0:
        return np.array([], dtype=np.int32)

    dists, idx = pmt_tree.query(H_POS, k=1)
    valid = dists < tol
    return np.unique(idx[valid]).astype(np.int32)

# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Caricamento modelli e scalers ────────────────────────────────────────
    model_hits = tf.keras.models.load_model("model.h5", compile=False)
    with open("scalers.pkl", "rb") as f:
        scalers_hits = pickle.load(f)

    model_lambda = tf.keras.models.load_model("model_lambda.h5", compile=False)
    with open("scalers_lambda.pkl", "rb") as f:
        scalers_lambda = pickle.load(f)

    sh_mean = tf.constant(
        [scalers_hits["dist"]["mean"], 0.0, 0.0,
         scalers_hits["time"]["mean"], scalers_hits["log_dist"]["mean"]],
        dtype=tf.float32,
    )
    sh_std = tf.constant(
        [scalers_hits["dist"]["std"], 1.0, 1.0,
         scalers_hits["time"]["std"], scalers_hits["log_dist"]["std"]],
        dtype=tf.float32,
    )
    sl_mean  = tf.constant(scalers_lambda.mean_,  dtype=tf.float32)
    sl_scale = tf.constant(scalers_lambda.scale_, dtype=tf.float32)



    n_events = len(events_)

    #events = [events_[4]]
    #n_events = 1
    events = events_

    if n_events == 0:
        raise RuntimeError(f"Nessun evento trovato")

    print(f"Caricati {n_events} eventi")
    print(f"Ricerca Angoli Pura (Posizione, Tempo ed Energia fissati a Ground Truth)")
    print(f"Likelihood mode        : {LIKELIHOOD_MODE}")
    print(f"Occupancy light scale  : {OCCUPANCY_LIGHT_SCALE:g}")
    print(f"Sigmoide Lambda (Fissa) : Soglia={LAMBDA_CUTOFF_THRESHOLD}, Damping={LAMBDA_CUTOFF_DAMPING}")

    # Accumulatori per le medie finali
    sum_dtheta = np.zeros((len(soglia_vals), len(k_ripidita_vals)))
    sum_dphi   = np.zeros((len(soglia_vals), len(k_ripidita_vals)))

    all_space_angles = np.zeros((len(soglia_vals), len(k_ripidita_vals), n_events))

    t0_global = time.time()

    for i_event, event in enumerate(events):
        p_true = np.asarray(event["p_true"], dtype=np.float64)

        # Creiamo i tensori in memoria una sola volta per questo evento
        H_POS_tf = tf.constant(np.asarray(event["H_POS"], dtype=np.float64).reshape(-1, 3), dtype=tf.float64)
        H_DIR_tf = tf.constant(np.asarray(event["H_DIR"], dtype=np.float64).reshape(-1, 3), dtype=tf.float64)
        H_T_tf   = tf.constant(np.asarray(event["H_T"], dtype=np.float64).reshape(-1), dtype=tf.float64)
        P_SIGNAL_tf = tf.constant(np.asarray(event["P_SIGNAL"], dtype=np.float64).reshape(-1), dtype=tf.float64)

        print("\n" + "=" * 70)
        print(f"EVENTO {i_event + 1}/{n_events}")
        print("Ground truth: " + format_ground_truth(p_true))
        print("-" * 70)
        #plot_topdown_view(p_true, P_POS, i_event)

        for i_soglia, soglia in enumerate(soglia_vals):
            for i_k, k_ripidita in enumerate(k_ripidita_vals):
                t_start = time.time()

                # Chiamiamo run_angle_search passandole i tensori dell'evento
                result = run_angle_search(
                    p_true,
                    H_POS_tf,
                    H_DIR_tf,
                    H_T_tf,
                    P_SIGNAL_tf,
                    float(soglia),
                    float(k_ripidita),
                )
                t_elapsed = time.time() - t_start

                all_space_angles[i_soglia, i_k, i_event] = result["space_angle_deg"]

                print(f"\nSoglia_geom={soglia:g} | k_geom={k_ripidita:g} | t={t_elapsed:.1f}s")
                print(f"  Conf. Intermedia | {_fmt(result['coarse_params'])}")
                print(f"  Conf. Finale     | {_fmt(result['fine_params'])}")
                print(f"  Scarto Spaziale  | α = {result['space_angle_deg']:>6.2f}°")


    # ── Risultati medi ───────────────────────────────────────────────────────
    total_time = time.time() - t0_global

    print("\n" + "=" * 70)
    print(f"TEMPO TOTALE per {n_events} eventi: {total_time:.2f} secondi")
    print("RISULTATI FINALI SU TUTTI GLI EVENTI (in gradi)")
    print("=" * 70)

    for i_soglia, soglia in enumerate(soglia_vals):
        for i_k, k_ripidita in enumerate(k_ripidita_vals):
            # Estraiamo l'array degli scarti per questa specifica configurazione
            scarti = all_space_angles[i_soglia, i_k, :]

            avg_alpha = np.mean(scarti)
            med_alpha = np.median(scarti)

            print(
                f"Soglia_geom={soglia:g} | k_geom={k_ripidita:g} | "
                f"Media = {avg_alpha:>6.2f}° | Mediana = {med_alpha:>6.2f}°"
            )
    print("=" * 70)

    # Ora ha senso una sola heatmap per la metrica globale
