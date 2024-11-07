import numpy as np


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
