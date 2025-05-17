
import numpy as np
import torch
from sklearn.metrics import r2_score
from hybrid_model_utils import (
    HybridMultiHead, reshape_signals, evaluate_model, plot_prediction_scatter2
)
from two_compartment_generator_vbonly import TwoCompartmentHPDataGenerator
import nibabel as nib

# === Generate synthetic dataset ===
time_points=np.arange(0, 60, 5) 
n_timepoints = len(time_points)
generator = TwoCompartmentHPDataGenerator(time_points=time_points)
n_samples = 50
X, y_true = generator.generate_dataset(n_samples=n_samples)  # shape: (N, T, 2)
X_norm, X_raw = reshape_signals(X)  
print("Generated shapes:", X.shape, y_true.shape, X_norm.shape, X_raw.shape)

min_value = np.min(X)
max_value = np.max(X)
print(f"The dataset has shape: {X.shape}")
print(f"The minimum value in the dataset is: {min_value}")
print(f"The maximum value in the dataset is: {max_value}")
print(f"The range of values in the dataset is: ({min_value}, {max_value})")


# === Load trained model ===
weights_path = r"output\demo_20250508-170410\trained_hybrid_model.pth"
input_dim_raw = X_raw.shape[1]
input_dim_norm = X_norm.shape[1]
model = HybridMultiHead(input_dim_raw=input_dim_raw, input_dim_norm=input_dim_norm)
model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
model.eval()


# === Predict ===
y_pred = evaluate_model(model, X_norm, X_raw, y_true)

# === Plotting ===
output_dir = "output"
import os
from datetime import datetime
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
fname = os.path.join(output_dir, f"NeuralNetworkFitOfGeneratorData_{timestamp}")
plot_prediction_scatter2(y_true, y_pred, saveName=fname)

from fit_two_compartment import fit_traditional_2c_model
from two_compartment_generator import TwoCompartmentHPDataGenerator

# Traditional fit comparison
generator_instance = TwoCompartmentHPDataGenerator(time_points=time_points)
traditional_preds = []

X_test = X

print("Using traditional fittings...")
import time
start_time = time.time()
for i in range(len(X_test)):
    pyr = X_test[i, :, 0]
    lac = X_test[i, :, 1]
    try:
        params, _, success = fit_traditional_2c_model(
            time_points, pyr, lac, generator_instance, estimate_r1=False)
        traditional_preds.append(params[:3])
    except:
        traditional_preds.append([np.nan, np.nan, np.nan])
tradfit_time = time.time() - start_time
print(f"Traditional model training time: {tradfit_time:.2f} seconds")
y_pred_fit = np.array(traditional_preds)

fname = os.path.join(output_dir, f"TraditionalFitOfGeneratorData_{timestamp}")
plot_prediction_scatter2(y_true, y_pred_fit, saveName=fname)


