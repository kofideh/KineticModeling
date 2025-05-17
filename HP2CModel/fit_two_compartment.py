# fit_two_compartment_fixed.py
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from two_compartment_generator import gamma_variate_aif
import traceback

def fit_traditional_2c_model(time_points, S_pyr_noisy, S_lac_noisy, generator_instance,
                             estimate_r1=False, initial_guess=None, bounds=None):
    combined_noisy_data = np.concatenate((S_pyr_noisy, S_lac_noisy))

    def deriv(y, t, kpl, kve, vb, r1p, r1l, aif_func):
        Pv, Pe, Le = y
        ut = aif_func(t)
        dPv_dt = kve * (Pe - Pv) - r1p * Pv + ut / vb
        dPe_dt = kve * vb / (1 - vb) * (Pv - Pe) - kpl * Pe - r1p * Pe
        dLe_dt = kpl * Pe - r1l * Le
        return [dPv_dt, dPe_dt, dLe_dt]

    def model_signal_wrapper(t, *params_to_fit):
        kpl, kve, vb = params_to_fit[0], params_to_fit[1], params_to_fit[2]
        if estimate_r1:
            r1p, r1l = params_to_fit[3], params_to_fit[4]
        else:
            # r1p = generator_instance.r1p_range[0]
            # r1l = generator_instance.r1l_range[0]
            r1p=1/30
            r1l=1/25

        if not (0 < vb < 1):
            return np.ones_like(combined_noisy_data) * 1e10

        try:
            aif_waveform = gamma_variate_aif(time_points, t0=10.0, alpha=2.0, beta=4.0)
            aif_func = interp1d(time_points, aif_waveform, kind='linear', fill_value=0.0, bounds_error=False)

            y0 = [0.0, 0.0, 0.0]
            sol = odeint(deriv, y0, t, args=(kpl, kve, vb, r1p, r1l, aif_func))
            Pv, Pe, Le = sol[:, 0], sol[:, 1], sol[:, 2]

            S_pyr = vb * Pv + (1 - vb) * Pe
            S_lac = (1 - vb) * Le
            max_signal = max(np.max(S_pyr), np.max(S_lac), 1e-9)
            S_pyr_norm = S_pyr / max_signal
            S_lac_norm = S_lac / max_signal

            return np.concatenate((S_pyr_norm, S_lac_norm))
        except Exception:
            return np.ones_like(combined_noisy_data) * 1e12

    if estimate_r1:
        n_params_fit = 5
        default_guess = [0.05, 0.1, 0.1, 1/30, 1/25]
        default_bounds = ([0, 0, 0.001, 0.01, 0.01], [0.5, 1.0, 0.5, 0.1, 0.1])
    else:
        n_params_fit = 3
        default_guess = [0.05, 0.1, 0.1]
        default_bounds = ([0, 0, 0.001], [0.5, 1.0, 0.5])

    if initial_guess is None:
        initial_guess = default_guess
    if bounds is None:
        bounds = default_bounds

    try:
        fitted_params, covariance = curve_fit(
            model_signal_wrapper,
            time_points,
            combined_noisy_data,
            p0=initial_guess,
            bounds=bounds,
            method='trf',
            maxfev=1000 * n_params_fit
        )
        std_devs = np.sqrt(np.diag(covariance))
        success = True
    except Exception as e:
        print(f"Fit failed: {e}")
        traceback.print_exc()
        fitted_params = np.array(initial_guess)
        std_devs = np.full_like(fitted_params, np.nan)
        success = False

    if not estimate_r1:
        full_params = np.concatenate((fitted_params, [generator_instance.r1p_range[0], generator_instance.r1l_range[0]]))
        full_std_devs = np.concatenate((std_devs, [0, 0]))
    else:
        full_params = fitted_params
        full_std_devs = std_devs

    return full_params, full_std_devs, success
