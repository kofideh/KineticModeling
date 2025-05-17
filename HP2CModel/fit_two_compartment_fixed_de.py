
# fit_two_compartment_fixed_de.py
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
from two_compartment_generator import gamma_variate_aif
import traceback

def fit_traditional_2c_model_de(time_points, S_pyr_noisy, S_lac_noisy, generator_instance,
                                estimate_r1=False):
    combined_noisy_data = np.concatenate((S_pyr_noisy, S_lac_noisy))

    def deriv(y, t, kpl, kve, vb, r1p, r1l, aif_func):
        Pv, Pe, Le = y
        ut = aif_func(t)
        dPv_dt = kve * (Pe - Pv) - r1p * Pv + ut / vb
        dPe_dt = kve * vb / (1 - vb) * (Pv - Pe) - kpl * Pe - r1p * Pe
        dLe_dt = kpl * Pe - r1l * Le
        return [dPv_dt, dPe_dt, dLe_dt]

    def model_signal(params):
        kpl, kve, vb = params[:3]
        if estimate_r1:
            r1p, r1l = params[3], params[4]
        else:
            r1p = 1 / 30
            r1l = 1 / 25

        if not (0 < vb < 1):
            return np.inf

        try:
            aif_waveform = gamma_variate_aif(time_points, t0=10.0, alpha=2.0, beta=4.0)
            aif_func = interp1d(time_points, aif_waveform, kind='linear', fill_value=0.0, bounds_error=False)

            y0 = [0.0, 0.0, 0.0]
            sol = odeint(deriv, y0, time_points, args=(kpl, kve, vb, r1p, r1l, aif_func))
            Pv, Pe, Le = sol[:, 0], sol[:, 1], sol[:, 2]

            S_pyr = vb * Pv + (1 - vb) * Pe
            S_lac = (1 - vb) * Le
            max_signal = max(np.max(S_pyr), np.max(S_lac), 1e-9)
            S_pyr_norm = S_pyr / max_signal
            S_lac_norm = S_lac / max_signal

            model_combined = np.concatenate((S_pyr_norm, S_lac_norm))
            return np.sum((combined_noisy_data - model_combined) ** 2)
        except Exception:
            return np.inf

    bounds = [(0, 0.5), (0, 1.0), (0.001, 0.5)]
    if estimate_r1:
        bounds.extend([(0.01, 0.1), (0.01, 0.1)])

    try:
        result = differential_evolution(model_signal, bounds, tol=1e-6, maxiter=1000)
        fitted_params = result.x
        success = result.success
    except Exception as e:
        print(f"Fit failed: {e}")
        traceback.print_exc()
        fitted_params = np.zeros(len(bounds))
        success = False

    if not estimate_r1:
        full_params = np.concatenate((fitted_params, [generator_instance.r1p_range[0], generator_instance.r1l_range[0]]))
    else:
        full_params = fitted_params

    return full_params, np.full_like(full_params, np.nan), success
