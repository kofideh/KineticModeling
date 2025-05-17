
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution

# === Two-compartment ODE system ===
def two_compartment_ode(y, t, kpl, kve, vb, R1p=1/30, R1l=1/25):
    P, L = y
    dPdt = -kpl * P - R1p * P
    dLdt = kpl * P - R1l * L
    return [dPdt, dLdt]

# === Signal model ===
def simulate_signals(t, kpl, kve, vb, R1p=1/30, R1l=1/25):
    P0 = 1.0  # normalized injected pyruvate
    y0 = [P0, 0.0]
    sol = odeint(two_compartment_ode, y0, t, args=(kpl, kve, vb, R1p, R1l))
    P, L = sol[:, 0], sol[:, 1]
    S_pyr = (1 - vb) * P + vb * P0 * np.exp(-t / 10)  # vascular + tissue
    S_lac = (1 - vb) * L
    return S_pyr, S_lac

# === Objective function ===
def two_compartment_residual(params, t, S_pyr_obs, S_lac_obs):
    kpl, kve, vb = params
    if not (0 <= kpl <= 0.5 and 0 <= kve <= 0.5 and 0 <= vb <= 1.0):
        return np.inf
    S_pyr_sim, S_lac_sim = simulate_signals(t, kpl, kve, vb)
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
    # Generate synthetic test case
    t = np.linspace(0, 60, 20)
    kpl_true, kve_true, vb_true = 0.1, 0.15, 0.2
    S_pyr, S_lac = simulate_signals(t, kpl_true, kve_true, vb_true)

    # Fit using DE
    kpl_fit, kve_fit, vb_fit = fit_two_compartment_de(t, S_pyr, S_lac)
    print(f"True: kpl={kpl_true}, kve={kve_true}, vb={vb_true}")
    print(f"Fitted: kpl={kpl_fit:.4f}, kve={kve_fit:.4f}, vb={vb_fit:.4f}")
