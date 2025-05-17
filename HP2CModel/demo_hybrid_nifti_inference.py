
import numpy as np
import torch
import nibabel as nib
from sklearn.metrics import r2_score
from glob import glob

from hybrid_model_utils import (
    HybridMultiHead, load_nifti_series, reshape_signals, preprocess_signals,
    evaluate_model, plot_prediction_scatter, plot_weight_histograms
)

# Re-run the preprocessing code after kernel reset
import numpy as np

def normalize_clinical_data(X_clinical):
    """
    Normalizes clinical input data by dividing each sample by its own peak value.
    
    Args:
        X_clinical (np.ndarray): shape (N, T, 2) — dynamic pyruvate and lactate signal
    
    Returns:
        X_norm (np.ndarray): normalized data of same shape
    """
    peak_vals = X_clinical.max(axis=1, keepdims=True) + 1e-8  # avoid division by zero
    return X_clinical / peak_vals

def normalize_clinical_data_with_peaks(X_clinical):
    peak_vals = X_clinical.max(axis=1, keepdims=True) + 1e-8
    return X_clinical / peak_vals, peak_vals


# === Example Setup ===
weights_path = r"output\demo_20250508-170410\trained_hybrid_model.pth"
# 1. Load trained model
input_dim_raw = 24  # e.g., 12 timepoints x 2 channels flattened
input_dim_norm = 24
model = HybridMultiHead(input_dim_raw=input_dim_raw, input_dim_norm=input_dim_norm)
model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
model.eval()

# 2. Load example NIfTI time series
# Suppose you have: [pyr_t0.nii.gz, ..., pyr_t11.nii.gz] and [lac_t0.nii.gz, ..., lac_t11.nii.gz]
# Concatenate into a (samples, timepoints, 2) numpy array
# Here's how you'd do it if using slices or ROI voxels from each image
# This is a mockup — replace with real filepaths
from glob import glob
# pyr_files = sorted(glob("data/pyr_t*.nii.gz"))
# lac_files = sorted(glob("data/lac_t*.nii.gz"))
pyr_files = sorted(glob("data/pyr*.nii.gz"))
lac_files = sorted(glob("data/lac*.nii.gz"))
# Assume files are pre-aligned and shapes match
pyr = load_nifti_series(pyr_files)  # shape: (X, Y, Z, T)
lac = load_nifti_series(lac_files)
pyr = pyr.squeeze()  # remove singleton dimensions if any
lac = lac.squeeze()


# === DENOISING ===
sigma = (1, 1, 1, 0)  # spatial smoothing only
from scipy.ndimage import gaussian_filter
pyr = gaussian_filter(pyr, sigma=sigma)
lac = gaussian_filter(lac, sigma=sigma)
print("Loaded shapes:", pyr.shape, lac.shape)

# Example: flatten to a 2D matrix of voxels × timepoints
pyr_2d = pyr.reshape(-1, pyr.shape[-1])
lac_2d = lac.reshape(-1, lac.shape[-1])
X_combined = np.stack([pyr_2d, lac_2d], axis=-1)  # shape: (voxels, timepoints, 2)
print("Combined shape:", X_combined.shape)

min_value = np.min(X_combined)
max_value = np.max(X_combined)
print(f"The range of values in the dataset is: ({min_value}, {max_value})")

# X_combined_norm = normalize_clinical_data(X_combined)
X_combined_norm = X_combined/np.max(X_combined)  # normalize by max value

min_value = np.min(X_combined_norm)
max_value = np.max(X_combined_norm)
print(f"The range of values in the normalized clinical dataset is: ({min_value}, {max_value})")

# 3. Preprocess: normalize for kpl/kve, raw for vb
X_norm, X_raw = reshape_signals(X_combined_norm)
# X_norm, X_raw = preprocess_signals(X_combined)
print("Generated shapes:", X_combined.shape, X_norm.shape, X_raw.shape)

# 4. Predict
pred = evaluate_model(model, X_norm, X_raw)

# Optional: compare with ground truth (if available)
# y_true = joblib.load("ground_truth_parameters.pkl")
# plot_prediction_scatter(y_true, pred)

# 5. Plot model weights
# plot_weight_histograms(model)
# Replace with the actual shape of your dynamic images (excluding time)
volume_shape = pyr.shape[:-1]  # e.g., (64, 64, 1)

# Save each parameter map
param_names = ["kPL", "kVE", "vB"]
for i, name in enumerate(param_names):
    param_map = pred[:, i].reshape(volume_shape)
    img = nib.Nifti1Image(param_map.astype(np.float32), affine=np.eye(4))
    nib.save(img, f"{name}_map_predicted.nii.gz")
