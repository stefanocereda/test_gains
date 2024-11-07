import numpy as np
import streamlit as st
import pandas as pd


its = np.linspace(1, 100, 100)
etas = [np.sqrt(8 * np.log(3) / it) for it in its]
st.write("# Eta as function of iteration")
st.line_chart(etas, x_label="iter", y_label="η")


def compute_probs(gains, it, norm=False):
    eta = np.sqrt(8 * np.log(3) / it)
    if norm:
        return compute_probs_eta_2(gains, eta)
    return compute_probs_eta(gains, eta)


def compute_probs_eta(gains, eta):
    logits = np.array(gains, dtype=float)
    logits -= np.max(logits)
    exp_logits = np.exp(eta * logits)
    probs = exp_logits / np.sum(exp_logits)
    return probs


def compute_probs_eta_2(gains, eta):
    logits = np.array(gains, dtype=float)
    logits -= np.max(logits)
    if any(g != 0 for g in gains):
        logits /= max(logits) - min(logits)
    exp_logits = np.exp(eta * logits)
    probs = exp_logits / np.sum(exp_logits)
    return probs


st.write("# Probabilities according to gains with iteration-based η")
g0 = st.slider("gain0", -10.0, 10.0, step=0.1)
g1 = st.slider("gain1", -10.0, 10.0, step=0.1)
g2 = st.slider("gain2", -10.0, 10.0, step=0.1)

gains = [g0, g1, g2]
all_probs = np.asarray([compute_probs(gains, it) for it in its])

chart_data = pd.DataFrame(
    {
        "iter": its,
        "prob0": all_probs[:, 0] * 100,
        "prob1": all_probs[:, 1] * 100,
        "prob2": all_probs[:, 2] * 100,
    }
)
st.line_chart(chart_data, x="iter", y_label="%")

st.write("# Probabilities with variable gains and optional normalization")
etas = np.linspace(0, 3, 100)
all_probs = np.asarray([compute_probs_eta(gains, eta) for eta in etas])
all_probs_2 = np.asarray([compute_probs_eta_2(gains, eta) for eta in etas])
chart_data = pd.DataFrame(
    {
        "eta": etas,
        "prob0": all_probs[:, 0] * 100,
        "prob1": all_probs[:, 1] * 100,
        "prob2": all_probs[:, 2] * 100,
        "prob0_2": all_probs_2[:, 0] * 100,
        "prob1_2": all_probs_2[:, 1] * 100,
        "prob2_2": all_probs_2[:, 2] * 100,
    }
)
st.line_chart(chart_data, x="eta", y_label="%")


st.write("# Example tuning")

st.write("## Settings")
col0, col1 = st.columns(2)
with col0:
    scale = st.slider("Noisiness", 0.0, 10.0, 1.0, step=0.01)
    history = st.slider("History", 0.0, 2.0, 1.0)
    seed = st.number_input("Random seed", value=0)
with col1:
    f0 = st.slider("Fitness af 0", -1.0, 1.0, -1.0, step=0.01)
    f1 = st.slider("Fitness af 1", -1.0, 1.0, 0.9, step=0.01)
    f2 = st.slider("Fitness af 2", -1.0, 1.0, 1.0, step=0.01)

np.random.seed(seed)
n_iters = 100
gains = np.zeros((n_iters + 1, 3))
gains_hist = np.zeros_like(gains)
probs_orig = np.zeros_like(gains)
probs_eta = np.zeros_like(gains)
probs_hist = np.zeros_like(gains)
probs_normhist = np.zeros_like(gains)

for it in range(n_iters):
    values = np.random.normal(loc=[-f0, -f1, -f2], scale=scale, size=3)
    gains[it + 1, :] = gains[it, :]
    gains[it + 1, :] -= values
    gains_hist[it + 1, :] = gains_hist[it, :] * history
    gains_hist[it + 1, :] -= values

    probs_orig[it + 1] = compute_probs_eta(gains[it + 1], 1)
    probs_eta[it + 1] = compute_probs(gains[it + 1], it + 1)
    probs_hist[it + 1] = compute_probs(gains_hist[it + 1], it + 1, False)
    probs_normhist[it + 1] = compute_probs(gains_hist[it + 1], it + 1, True)

st.write("## Gain evolution")
cols = st.columns(2)
for col, ga, name in zip(cols, [gains, gains_hist], ["as-is", "weighted-history"]):
    with col:
        st.write(name)
        chart_data = pd.DataFrame(
            {
                "gains0": ga[:, 0],
                "gains1": ga[:, 1],
                "gains2": ga[:, 2],
            }
        )
        st.line_chart(chart_data, x_label="Iteration", y_label="Gain")

st.write("## Probability evolution")
cols = st.columns(4)
for col, prob, name in zip(
    cols,
    [probs_orig, probs_eta, probs_hist, probs_normhist],
    ["as-is", "variable η", "η+history", "η+history+norm"],
):
    with col:
        st.write(name)
        chart_data = pd.DataFrame(
            {
                "prob0": prob[:, 0],
                "prob1": prob[:, 1],
                "prob2": prob[:, 2],
            }
        )
        st.line_chart(chart_data)
