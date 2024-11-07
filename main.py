import numpy as np
import streamlit as st
import pandas as pd
from functions import *


st.set_page_config("wide")


st.write("# Example tuning")

st.write("## Settings")
with st.sidebar:
    scale = st.slider("Noisiness", 0.0, 10.0, 1.0, step=0.01)
    history = st.slider("History", 0.0, 2.0, 1.0)
    seed = st.number_input("Random seed", value=0)
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
