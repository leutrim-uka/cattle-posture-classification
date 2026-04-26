import numpy as np
import pandas as pd

EPS = 1e-12
LOG2PI = np.log(2.0 * np.pi)

def logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))

def make_transmat(p00: float, p11: float) -> np.ndarray:
    p00 = float(np.clip(p00, 1e-6, 1.0 - 1e-6))
    p11 = float(np.clip(p11, 1e-6, 1.0 - 1e-6))
    return np.array([[p00, 1.0 - p00],
                     [1.0 - p11, p11]], dtype=float)

def gaussian_logpdf_diag(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    # x shape (T,), returns shape (T,)
    var = float(max(var, 1e-9))
    return -0.5 * (LOG2PI + np.log(var) + ((x - mean) ** 2) / var)

def emission_stats_from_val(probs_val, y_val):
    z = logit(np.asarray(probs_val).ravel())
    y = np.asarray(y_val).astype(int).ravel()

    z0 = z[y == 0]
    z1 = z[y == 1]
    # guard against empty slices
    if len(z0) == 0 or len(z1) == 0:
        raise ValueError("Validation split has only one class; cannot fit emission stats.")

    mu0, v0 = float(z0.mean()), float(z0.var() + 1e-6)
    mu1, v1 = float(z1.mean()), float(z1.var() + 1e-6)
    means = np.array([mu0, mu1], dtype=float)
    vars_ = np.array([v0, v1], dtype=float)
    return means, vars_

def viterbi_decode_with_gaps(probs, timestamps, startprob, transmat_1min, means, vars_):
    """
    Gap-aware 2-state Viterbi.
    - timestamps may be irregular
    - transition between t-1 and t uses A^(gap_minutes)
    Returns decoded states aligned to the input order (after sorting by time).
    """
    probs = np.asarray(probs).ravel()
    ts = pd.to_datetime(timestamps)

    order = np.argsort(ts.to_numpy())
    probs = probs[order]
    ts = ts.iloc[order] if isinstance(ts, pd.Series) else ts[order]

    z = logit(probs).astype(float)  # shape (T,)
    T = len(z)

    # emission log-likelihoods: shape (T, 2)
    emit = np.zeros((T, 2), dtype=float)
    emit[:, 0] = gaussian_logpdf_diag(z, means[0], vars_[0])
    emit[:, 1] = gaussian_logpdf_diag(z, means[1], vars_[1])

    # time gaps in minutes (int), first gap unused
    dt = np.diff(pd.to_datetime(ts).to_numpy()).astype("timedelta64[m]").astype(int)
    dt = np.maximum(dt, 1)  # if duplicates or weird ordering, treat as 1

    # Viterbi DP
    log_start = np.log(np.clip(startprob, EPS, 1.0))
    delta = np.full((T, 2), -np.inf, dtype=float)
    psi = np.zeros((T, 2), dtype=int)

    delta[0] = log_start + emit[0]

    for t in range(1, T):
        k = int(dt[t - 1])
        A_k = np.linalg.matrix_power(transmat_1min, k)
        logA = np.log(np.clip(A_k, EPS, 1.0))

        for j in (0, 1):
            scores = delta[t - 1] + logA[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = emit[t, j] + float(np.max(scores))

    # backtrack
    states = np.zeros(T, dtype=int)
    states[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    # map back to original ordering
    states_unsorted = np.empty_like(states)
    states_unsorted[order] = states
    return states_unsorted

def hmm_smooth_per_session_gapaware(probs, sessions, timestamps, params):
    probs = np.asarray(probs).ravel()
    sessions = np.asarray(sessions)
    timestamps = pd.to_datetime(timestamps)

    out = np.zeros_like(probs, dtype=int)

    for sid in np.unique(sessions):
        idx = np.where(sessions == sid)[0]
        if len(idx) == 0:
            continue

        states = viterbi_decode_with_gaps(
            probs=probs[idx],
            timestamps=timestamps.iloc[idx] if isinstance(timestamps, pd.Series) else timestamps[idx],
            startprob=params["startprob"],
            transmat_1min=params["transmat"],
            means=params["means"],
            vars_=params["vars"],
        )
        out[idx] = states

    return out

def tune_transmat_on_val(val_probs, val_labels, val_sessions, val_timestamps,
                         startprob, means, vars_,
                         grid=(0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998)):
    """
    Tunes per-minute A by selecting (p00, p11) that yields best macro F1 on VAL after smoothing.
    Uses only VAL cows (already group-held-out), so no test leakage.
    """
    from sklearn.metrics import f1_score

    y_val = np.asarray(val_labels).astype(int).ravel()
    best = {"score": -np.inf, "p00": None, "p11": None}

    for p00 in grid:
        for p11 in grid:
            A = make_transmat(p00, p11)
            params = {"startprob": startprob, "transmat": A, "means": means, "vars": vars_}
            yhat = hmm_smooth_per_session_gapaware(
                probs=val_probs,
                sessions=np.asarray(val_sessions),
                timestamps=pd.to_datetime(val_timestamps),
                params=params,
            )
            score = f1_score(y_val, yhat, average="macro")
            if score > best["score"]:
                best = {"score": score, "p00": p00, "p11": p11}

    return make_transmat(best["p00"], best["p11"])

def fit_hmm_params_with_val_tuning(train_labels, val_probs, val_labels, val_sessions, val_timestamps):
    train_labels = np.asarray(train_labels).astype(int).ravel()
    startprob = np.array([1.0 - train_labels.mean(), train_labels.mean()], dtype=float)

    means, vars_ = emission_stats_from_val(val_probs, val_labels)

    # Tune per-minute transmat on VAL sequences
    transmat_1min = tune_transmat_on_val(
        val_probs=val_probs,
        val_labels=val_labels,
        val_sessions=val_sessions,
        val_timestamps=val_timestamps,
        startprob=startprob,
        means=means,
        vars_=vars_,
    )

    return {"startprob": startprob, "transmat": transmat_1min, "means": means, "vars": vars_}

def build_session_ids(df, animal_col, time_col, max_gap_minutes=5):
    g = df[[animal_col, time_col]].copy()
    g[time_col] = pd.to_datetime(g[time_col])
    g = g.sort_values([animal_col, time_col])
    gap = g.groupby(animal_col)[time_col].diff().dt.total_seconds()
    local_sid = (gap > max_gap_minutes * 60).groupby(g[animal_col]).cumsum().fillna(0).astype(int)
    sid = g[animal_col].astype(str) + "_" + local_sid.astype(str)
    # align back to original index order
    return sid.reindex(df.sort_values([animal_col, time_col]).index).reindex(df.index)