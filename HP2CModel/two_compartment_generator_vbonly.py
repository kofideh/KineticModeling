import numpy as np
from scipy.integrate import odeint

def generate_variable_aif_params():
    return {'t0': 0, 'alpha': 3.0, 'beta': 1.0}

class TwoCompartmentHPDataGenerator:
    def __init__(self, vb_range=(0.01, 0.5), kpl_range=(0.01, 0.2), kve_range=(0.05, 0.45),
                 r1p=1/30, r1l=1/25, time_points=np.linspace(0, 60, 12)):
        self.vb_range = vb_range
        self.kpl_range = kpl_range
        self.kve_range = kve_range
        self.r1p = r1p
        self.r1l = r1l
        self.time_points = time_points

    def _sample_params(self):
        return {
            'vb': np.random.uniform(*self.vb_range),
            'kpl': np.random.uniform(*self.kpl_range),
            'kve': np.random.uniform(*self.kve_range),
            'r1p': self.r1p,
            'r1l': self.r1l,
            **generate_variable_aif_params()
        }

    def _aif(self, t, t0, alpha, beta):
        t_shifted = np.maximum(t - t0, 0)
        return alpha * t_shifted * np.exp(-beta * t_shifted)

    def _solve_2c_model(self, params):
        def deriv(y, t):
            Pe, Le = y
            AIF = self._aif(t, params['t0'], params['alpha'], params['beta'])
            dPe_dt = AIF - (params['kpl'] + params['kve'] + params['r1p']) * Pe
            dLe_dt = params['kpl'] * Pe - params['r1l'] * Le
            return [dPe_dt, dLe_dt]

        y0 = [0, 0]
        sol = odeint(deriv, y0, self.time_points)
        Pe, Le = sol[:, 0], sol[:, 1]
        Pv = self._aif(self.time_points, params['t0'], params['alpha'], params['beta'])
        S_pyr = params['vb'] * Pv + (1 - params['vb']) * Pe
        S_lac = (1 - params['vb']) * Le
        return {'S_pyr': S_pyr, 'S_lac': S_lac, 'time': self.time_points}

    def generate_dataset(self, n_samples=1000, noise_std=0.05):
        X = []
        y = []
        for _ in range(n_samples):
            params = self._sample_params()
            result = self._solve_2c_model(params)
            S_pyr = result['S_pyr'] + np.random.normal(0, noise_std, size=len(result['S_pyr']))
            S_lac = result['S_lac'] + np.random.normal(0, noise_std, size=len(result['S_lac']))
            X.append(np.stack([S_pyr, S_lac], axis=1))  # shape: (time, 2)
            y.append([params['kpl'], params['kve'], params['vb']])  # exclude r1p and r1l
        return np.array(X), np.array(y)
