import numpy as np
import streamlit as st
import pandas as pd
from functions import *


its = np.linspace(1, 100, 100)
etas = [np.sqrt(8 * np.log(3) / it) for it in its]
st.write("# Eta as function of iteration")
st.line_chart(etas, x_label="iter", y_label="η")


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
