import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d # Added for AIF interpolation
import pandas as pd

# --- Helper Functions for Variable AIF ---
def generate_variable_aif_params(t0_range=(5.0, 15.0),
                                 alpha_range=(1.5, 4.0),
                                 beta_range=(2.0, 8.0)):
    """
    Samples random AIF parameters from specified uniform ranges.

    Args:
        t0_range (tuple): Min and max arrival time (seconds).
        alpha_range (tuple): Min and max shape parameter.
        beta_range (tuple): Min and max scale parameter (seconds).

    Returns:
        dict: Dictionary containing the sampled 't0', 'alpha', and 'beta'.
    """
    t0 = np.random.uniform(t0_range[0], t0_range[1])
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    beta = np.random.uniform(beta_range[0], beta_range[1])

    # Basic sanity checks
    if alpha <= 0 or beta <= 0:
        print(f"Warning: Sampled non-positive alpha ({alpha}) or beta ({beta}). Check ranges.")
        alpha = max(alpha, 1e-6)
        beta = max(beta, 1e-6)

    return {'t0': t0, 'alpha': alpha, 'beta': beta}


def gamma_variate_aif(time_vec, t0, alpha, beta, normalize_peak=True):
    """
    Calculates the Arterial Input Function (AIF) using the gamma-variate model.

    AIF(t) = A * (t - t0)**alpha * exp(-(t - t0) / beta) for t > t0

    Args:
        time_vec (np.ndarray): Array of time points (seconds).
        t0 (float): Arrival time delay (seconds).
        alpha (float): Shape parameter (dimensionless, > 0).
        beta (float): Scale parameter (seconds, > 0).
        normalize_peak (bool): If True, scale the AIF so its peak value is 1.0.
                               If False, A=1 is used (unnormalized amplitude).

    Returns:
        np.ndarray: The calculated AIF values corresponding to time_vec.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive.")

    delta_t = time_vec - t0
    delta_t_safe = np.maximum(delta_t, 1e-9)

    term = (delta_t_safe**alpha) * np.exp(-delta_t_safe / beta)
    aif_unscaled = np.where(delta_t > 0, term, 0.0)

    if normalize_peak:
        delta_t_peak = alpha * beta
        peak_val_unscaled = (delta_t_peak**alpha) * np.exp(-alpha)

        if peak_val_unscaled > 1e-12:
            A = 1.0 / peak_val_unscaled
        else:
            A = 1.0
            # print(f"Warning: Unscaled peak value is near zero ({peak_val_unscaled}). Normalization might be inaccurate.")

        aif = A * aif_unscaled
    else:
        aif = aif_unscaled

    return aif

# --- Main Generator Class ---
class TwoCompartmentHPDataGenerator:
    """
    Generates synthetic hyperpolarized 13C MRI time-series data
    based on a 2-compartment (vascular/extravascular) exchange model,
    now with a variable gamma-variate AIF.

    Compartments:
    - Vascular (v): Pyruvate (Pv)
    - Extravascular (e): Pyruvate (Pe), Lactate (Le)

    Observable Signals:
    - S_pyr = vb * Pv + (1 - vb) * Pe
    - S_lac = (1 - vb) * Le
    """

    def __init__(self,
                 time_points=None,
                 kpl_range=(0.01, 0.2),      # Pyruvate to Lactate conversion (in tissue) [s^-1]
                 kve_range=(0.05, 0.4),      # Pyruvate vascular extravasation rate [s^-1]
                 vb_range=(0.02, 0.15),       # Fractional vascular blood volume [unitless]
                 r1p_range=(1/30, 1/30),     # Pyruvate relaxation rate [s^-1]
                 r1l_range=(1/25, 1/25),     # Lactate relaxation rate [s^-1]
                 t0_range=(5.0-2, 15.0+2),       # AIF arrival time range [s]
                 alpha_range=(1.5-1.5, 4.0+2),     # AIF shape parameter range
                 beta_range=(2.0-2, 8.0+2),      # AIF scale parameter range [s]
                 noise_level_range=(0.02, 0.08), # Range for additive Gaussian noise std dev
                 snr_target_range=None,     # Optional: Target SNR range (alternative to noise_level)
                 theta_pyr=11,               # Flip angle for pyruvate (degrees)
                 theta_lac=80,               # Flip angle for lactate (degrees)
                 TR=5                        # Repetition time (seconds)
                 ):

        if time_points is None:
            self.time_points = np.arange(0, 60 + TR, TR) # Default time vector
        else:
            self.time_points = np.asarray(time_points)

        # Store parameter ranges
        self.kpl_range = kpl_range
        self.kve_range = kve_range
        self.vb_range = vb_range
        self.r1p_range = r1p_range
        self.r1l_range = r1l_range
        self.t0_range = t0_range
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.noise_level_range = noise_level_range
        self.snr_target_range = snr_target_range

        # Flip angles and TR
        self.theta_pyr_rad = np.deg2rad(theta_pyr)
        self.theta_lac_rad = np.deg2rad(theta_lac)
        self.TR = TR

        # Calculate apparent relaxation rates including RF effects
        # R1app = R1 - log(cos(theta))/TR
        # Ensure ranges are tuples/lists
        r1p_min, r1p_max = self.r1p_range if isinstance(self.r1p_range, (list, tuple)) else (self.r1p_range, self.r1p_range)
        r1l_min, r1l_max = self.r1l_range if isinstance(self.r1l_range, (list, tuple)) else (self.r1l_range, self.r1l_range)

        # Note: R1app calculation is now done inside _solve_2c_model
        # using the *sampled* R1 values for that specific simulation.

    def _sample_params(self):
        """Samples a set of model parameters from the specified ranges."""
        params = {
            'kpl': np.random.uniform(*self.kpl_range),
            'kve': np.random.uniform(*self.kve_range),
            'vb': np.random.uniform(*self.vb_range),
            'r1p': np.random.uniform(*self.r1p_range) if isinstance(self.r1p_range, (list, tuple)) else self.r1p_range,
            'r1l': np.random.uniform(*self.r1l_range) if isinstance(self.r1l_range, (list, tuple)) else self.r1l_range,
        }
        # Sample AIF parameters
        aif_params = generate_variable_aif_params(self.t0_range, self.alpha_range, self.beta_range)
        params.update(aif_params) # Add t0, alpha, beta to the dictionary

        return params

    def _solve_2c_model(self, params):
        """Solves the 2-compartment model ODEs for a given set of parameters."""

        # Unpack parameters
        kpl = params['kpl']
        kve = params['kve']
        vb = params['vb']
        r1p = params['r1p']
        r1l = params['r1l']
        t0 = params['t0']
        alpha = params['alpha']
        beta = params['beta']

        # Calculate apparent relaxation rates using sampled R1 and flip angles
        # Using log(cos()) might lead to issues if cos(theta) is negative (shouldn't happen for typical angles)
        # Ensure cos is positive
        cos_theta_pyr = np.cos(self.theta_pyr_rad)
        cos_theta_lac = np.cos(self.theta_lac_rad)

        if cos_theta_pyr <= 0 or cos_theta_lac <= 0:
             raise ValueError("Flip angles result in non-positive cosine, cannot calculate R1app.")

        r1app_p = r1p - np.log(cos_theta_pyr) / self.TR
        r1app_l = r1l - np.log(cos_theta_lac) / self.TR

        # Generate the AIF waveform for this specific sample
        aif_waveform = gamma_variate_aif(self.time_points, t0, alpha, beta, normalize_peak=True)

        # Create an interpolation function for the AIF
        # This allows odeint to get the AIF value at any intermediate time point t
        try:
            aif_interp = interp1d(self.time_points, aif_waveform, kind='linear', bounds_error=False, fill_value=0.0)
        except ImportError:
             print("Warning: scipy.interpolate.interp1d not found. ODE solver might be less accurate.")
             # Fallback: use nearest neighbor or simple lookup if interp1d is unavailable
             aif_interp = lambda t: aif_waveform[np.argmin(np.abs(self.time_points - t))]


        # Define the derivative function for the ODE solver
        def deriv(y, t, kpl, kve, vb, r1app_p, r1app_l, aif_func):
            Pv, Pe, Le = y
            # Get AIF value at time t using interpolation
            ut = aif_func(t)

            dPv_dt = kve * (Pe - Pv) - r1app_p * Pv + ut / vb # AIF input scaled by Vb
            dPe_dt = kve * vb / (1 - vb) * (Pv - Pe) - kpl * Pe - r1app_p * Pe
            dLe_dt = kpl * Pe - r1app_l * Le
            return [dPv_dt, dPe_dt, dLe_dt]

        # Initial conditions (assuming no signal before agent arrival)
        y0 = [0.0, 0.0, 0.0]

        # Solve the ODE system
        # Use finer time resolution for ODE solver for stability, then sample at self.time_points
        t_eval_fine = np.linspace(self.time_points.min(), self.time_points.max(), num=len(self.time_points) * 5)
        sol = odeint(deriv, y0, t_eval_fine, args=(kpl, kve, vb, r1app_p, r1app_l, aif_interp))

        # Interpolate solution back to the desired self.time_points
        sol_interp_func_pv = interp1d(t_eval_fine, sol[:, 0], kind='linear', bounds_error=False, fill_value=0.0)
        sol_interp_func_pe = interp1d(t_eval_fine, sol[:, 1], kind='linear', bounds_error=False, fill_value=0.0)
        sol_interp_func_le = interp1d(t_eval_fine, sol[:, 2], kind='linear', bounds_error=False, fill_value=0.0)

        Pv_sampled = sol_interp_func_pv(self.time_points)
        Pe_sampled = sol_interp_func_pe(self.time_points)
        Le_sampled = sol_interp_func_le(self.time_points)

        # Calculate observable signals
        S_pyr = vb * Pv_sampled + (1 - vb) * Pe_sampled
        S_lac = (1 - vb) * Le_sampled

        # Store results in a dictionary or DataFrame
        results = pd.DataFrame({
            'time': self.time_points,
            'AIF': aif_waveform, # Store the generated AIF waveform
            'Pv': Pv_sampled,
            'Pe': Pe_sampled,
            'Le': Le_sampled,
            'S_pyr': S_pyr,
            'S_lac': S_lac
        })

        return results

    def add_noise(self, clean_signal):
        """Adds Gaussian noise to the signal."""
        if self.snr_target_range:
             # Calculate noise based on target SNR
             snr_db = np.random.uniform(*self.snr_target_range)
             snr_linear = 10**(snr_db / 10.0)
             signal_power = np.mean(clean_signal**2)
             if signal_power < 1e-12: # Avoid division by zero for null signals
                 noise_power = 0
             else:
                noise_power = signal_power / snr_linear
             noise_std = np.sqrt(noise_power)
        else:
            # Calculate noise based on fixed noise level range (relative to max signal)
            noise_level = np.random.uniform(*self.noise_level_range)
            signal_max = np.max(np.abs(clean_signal))
            if signal_max < 1e-9: # Avoid large noise for near-zero signals
                noise_std = 0
            else:
                noise_std = noise_level * signal_max

        noise = np.random.normal(0, noise_std, clean_signal.shape)
        return clean_signal + noise

    def generate_sample(self):
        """Generates a single noisy data sample."""
        params = self._sample_params()
        clean_data = self._solve_2c_model(params)

        # Add noise to observable signals
        noisy_s_pyr = self.add_noise(clean_data['S_pyr'].values)
        noisy_s_lac = self.add_noise(clean_data['S_lac'].values)

        noisy_data = pd.DataFrame({
            'time': self.time_points,
            'S_pyr_noisy': noisy_s_pyr,
            'S_lac_noisy': noisy_s_lac
        })

        # Combine clean signal info with noisy signal for reference if needed
        full_data = pd.merge(clean_data, noisy_data, on='time')

        return params, full_data

    def generate_dataset(self, n_samples, estimate_r1=False):
        """
        Generates a dataset of n_samples.

        Args:
            n_samples (int): Number of samples to generate.
            estimate_r1 (bool): If True, include r1p and r1l in the target parameters y.
                                If False, only include kpl, kve, vb.

        Returns:
            tuple: (X, y, param_names)
                   X (np.ndarray): Input data array (n_samples, n_timepoints, 2 metabolites).
                   y (np.ndarray): Target parameter array (n_samples, n_params).
                   param_names (list): List of names for the target parameters in y.
        """
        X_list = []
        y_list = []

        target_param_keys = ['kpl', 'kve', 'vb']
        if estimate_r1:
            target_param_keys.extend(['r1p', 'r1l'])
        param_names = target_param_keys

        for _ in range(n_samples):
            params, data = self.generate_sample()

            # Stack noisy signals for network input (timepoints, metabolites)
            sample_x = np.stack([data['S_pyr_noisy'].values, data['S_lac_noisy'].values], axis=-1)

            # Select target parameters based on estimate_r1 flag
            sample_y = [params[key] for key in target_param_keys]

            X_list.append(sample_x)
            y_list.append(sample_y)

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y, param_names

    def plot_example(self):
        """Generates and plots a single example simulation."""
        params, data = self.generate_sample()

        plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        # Plot 1: AIF
        axes[0].plot(data['time'], data['AIF'], 'r-', label='Generated AIF')
        axes[0].set_ylabel('AIF Amplitude')
        axes[0].set_title('Arterial Input Function (AIF)')
        axes[0].legend()
        axes[0].grid(True)

        # Plot 2: Observable Signals (Clean vs Noisy)
        axes[1].plot(data['time'], data['S_pyr'], 'b--', alpha=0.7, label='S_pyr (Clean)')
        axes[1].plot(data['time'], data['S_lac'], 'g--', alpha=0.7, label='S_lac (Clean)')
        axes[1].plot(data['time'], data['S_pyr_noisy'], 'b-', marker='.', linestyle='None', markersize=5, label='S_pyr (Noisy)')
        axes[1].plot(data['time'], data['S_lac_noisy'], 'g-', marker='.', linestyle='None', markersize=5, label='S_lac (Noisy)')
        axes[1].set_ylabel('Observable Signal')
        axes[1].set_title('Observable Signals (Tissue Average)')
        axes[1].legend()
        axes[1].grid(True)

        # Plot 3: Underlying Compartment Signals (Clean)
        axes[2].plot(data['time'], data['Pv'], 'r-', label='Pv (Vascular)')
        axes[2].plot(data['time'], data['Pe'], 'm-', label='Pe (Extravascular)')
        axes[2].plot(data['time'], data['Le'], 'c-', label='Le (Extravascular)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Normalized Signal')
        axes[2].set_title('Underlying Compartment Signals (Clean)')
        axes[2].legend()
        axes[2].grid(True)

        # Construct title with all parameters
        title_str = (f'Example 2C Simulation\n'
                     f'kPL={params["kpl"]:.3f}, kve={params["kve"]:.3f}, vb={params["vb"]:.3f}, '
                     f'R1p={params["r1p"]:.3f}, R1l={params["r1l"]:.3f}\n'
                     f't0={params["t0"]:.1f}, alpha={params["alpha"]:.1f}, beta={params["beta"]:.1f}')
        plt.suptitle(title_str)
        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust layout for suptitle
        plt.show()

        return params, data # Return data instead of clean/noisy separately

# Example usage:
if __name__ == "__main__":
    generator_2c_variable_aif = TwoCompartmentHPDataGenerator(
        r1p_range=(0.02, 0.04), # Example with R1 variation
        r1l_range=(0.02, 0.04),
        # Using default variable AIF ranges
        time_points=np.arange(0, 70, 2) # Example with different time points
    )

    print("Generator initialized with variable AIF.")

    # Plot an example
    print("\nPlotting example simulation...")
    example_params, example_data = generator_2c_variable_aif.plot_example()

    print("\nExample Generated Parameters:")
    for key, value in example_params.items():
        print(f"  {key}: {value:.4f}")

    # Generate a small dataset
    print("\nGenerating small dataset (N=10)...")
    X_data, y_data, param_names_out = generator_2c_variable_aif.generate_dataset(10, estimate_r1=True)
    print(f"Dataset shapes: X={X_data.shape}, y={y_data.shape}")
    print(f"Target parameters: {param_names_out}")
    print("First sample parameters (y[0]):", y_data[0])