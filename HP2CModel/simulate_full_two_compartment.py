
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution

# === Full two-compartment ODE system ===
def full_two_compartment_ode(y, t, kpl, kve, vb, R1p=1/30, R1l=1/25, R1v=1/20):
    P_v, P_e, L = y
    dP_v_dt = -kve * (P_v - P_e) - R1v * P_v
    dP_e_dt = kve * (P_v - P_e) - kpl * P_e - R1p * P_e
    dL_dt = kpl * P_e - R1l * L
    return [dP_v_dt, dP_e_dt, dL_dt]

# === Gamma-variate arterial input function (AIF) ===
def aif(t, A=1.0, t0=10, alpha=2, beta=4):
    t_shifted = t - t0
    t_shifted[t_shifted < 0] = 0
    return A * (t_shifted ** alpha) * np.exp(-t_shifted / beta)

# === Simulate signals from full two-compartment model ===
def simulate_full_signals(t, kpl, kve, vb, R1p=1/30, R1l=1/25, R1v=1/20):
    dt = t[1] - t[0]
    aif_values = aif(t)
    P_v_input = np.interp(t, t, aif_values)
    y = np.zeros((len(t), 3))
    y[0, :] = [0, 0, 0]

    for i in range(1, len(t)):
        args = (kpl, kve, vb, R1p, R1l, R1v)
        y[i] = odeint(full_two_compartment_ode, y[i-1], [t[i-1], t[i]], args=args)[-1]
        y[i, 0] += P_v_input[i] * dt / vb  # infusion into vascular compartment

    P_v, P_e, L = y[:, 0], y[:, 1], y[:, 2]
    S_pyr = vb * P_v + (1 - vb) * P_e
    S_lac = (1 - vb) * L
    return S_pyr, S_lac

# === Objective function ===
def two_compartment_residual(params, t, S_pyr_obs, S_lac_obs):
    kpl, kve, vb = params
    if not (0 <= kpl <= 0.5 and 0 <= kve <= 0.5 and 0 <= vb <= 1.0):
        return np.inf
    S_pyr_sim, S_lac_sim = simulate_full_signals(t, kpl, kve, vb)
    return np.sum((S_pyr_obs - S_pyr_sim) ** 2 + (S_lac_obs - S_lac_sim) ** 2)

# === Fit function ===
def fit_two_compartment_de(t, S_pyr_obs, S_lac_obs):
    bounds = [(1e-4, 0.5), (1e-4, 0.5), (0.0, 1.0)]  # kpl, kve, vb
    result = differential_evolution(
        two_compartment_residual, bounds, args=(t, S_pyr_obs, S_lac_obs),
        strategy='best1bin', maxiter=200, popsize=15, tol=1e-6, disp=False
    )
    return result.x if result.success else [np.nan, np.nan, np.nan]

# === Example usage ===
if __name__ == "__main__":
    t = np.linspace(0, 60, 20)
    kpl_true, kve_true, vb_true = 0.1, 0.2, 0.15
    S_pyr, S_lac = simulate_full_signals(t, kpl_true, kve_true, vb_true)
    print("S_pyr:", S_pyr)
    print("S_lac:", S_lac)
    
     # Fit using DE
    kpl_fit, kve_fit, vb_fit = fit_two_compartment_de(t, S_pyr, S_lac)
    print(f"True: kpl={kpl_true}, kve={kve_true}, vb={vb_true}")
    print(f"Fitted: kpl={kpl_fit:.4f}, kve={kve_fit:.4f}, vb={vb_fit:.4f}")
