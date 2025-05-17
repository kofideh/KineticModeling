import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from two_compartment_generator_vbonly import TwoCompartmentHPDataGenerator
import os
from datetime import datetime
from hybrid_model_utils import HybridMultiHead

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(output_dir, f"demo_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# 1. Generate data
time_points=np.arange(0, 60, 5) 
n_timepoints = len(time_points)
n_samples = 100*10
generator = TwoCompartmentHPDataGenerator(time_points=time_points)
X, y = generator.generate_dataset(n_samples=n_samples, noise_std=0.05)

# 2. Split channels
X_raw = X.copy()  # raw data for vb
X_max = X.max(axis=1, keepdims=True) + 1e-6
# X_norm = X / X_max  # normalized data for kpl and kve
X_norm = X 
# 3. Flatten inputs
X_raw_flat = X_raw.reshape(X.shape[0], -1)
X_norm_flat = X_norm.reshape(X.shape[0], -1)

Xr_train, Xr_temp, Xn_train, Xn_temp, y_train, y_temp = train_test_split(
    X_raw_flat, X_norm_flat, y, test_size=0.3, random_state=42)
Xr_val, Xr_test, Xn_val, Xn_test, y_val, y_test = train_test_split(
    Xr_temp, Xn_temp, y_temp, test_size=0.05, random_state=42)


# 5. Convert to tensors
Xr_train_tensor = torch.tensor(Xr_train, dtype=torch.float32)
Xn_train_tensor = torch.tensor(Xn_train, dtype=torch.float32)
Xr_val_tensor = torch.tensor(Xr_val, dtype=torch.float32)
Xn_val_tensor = torch.tensor(Xn_val, dtype=torch.float32)
Xr_test_tensor = torch.tensor(Xr_test, dtype=torch.float32)
Xn_test_tensor = torch.tensor(Xn_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


model = HybridMultiHead(input_dim_raw=X_raw_flat.shape[1], input_dim_norm=X_norm_flat.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-35)

# 7. Training loop
n_epochs = 100*20
print("Training Neural Network...")
import time
start_time = time.time()
for epoch in range(n_epochs):
    model.train()
    preds = model(Xn_train_tensor, Xr_train_tensor)
    loss = criterion(preds, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(Xn_val_tensor, Xr_val_tensor)
        val_loss = criterion(val_preds, y_val_tensor)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

model_path = os.path.join(run_dir, "trained_hybrid_positive.pth")
# torch.save(model.state_dict(), model_path)
# After training
torch.save(model.state_dict(), "trained_hybrid_positive.pth")
print("Saved model with positivity constraints to trained_hybrid_positive.pth")
training_time = time.time() - start_time
# 8. Evaluation
model.eval()

r2_kpl = 0
r2_kve = 0
r2_vb  = 0


with torch.no_grad():
    y_pred = model(Xn_test_tensor, Xr_test_tensor).numpy()
    y_true = y_test_tensor.numpy()

    r2_kpl = 1 - np.sum((y_pred[:, 0] - y_true[:, 0])**2) / np.sum((y_true[:, 0] - np.mean(y_true[:, 0]))**2)
    r2_kve = 1 - np.sum((y_pred[:, 1] - y_true[:, 1])**2) / np.sum((y_true[:, 1] - np.mean(y_true[:, 1]))**2)
    r2_vb  = 1 - np.sum((y_pred[:, 2] - y_true[:, 2])**2) / np.sum((y_true[:, 2] - np.mean(y_true[:, 2]))**2)

    # Plot true vs. predicted for neural network
    import matplotlib.pyplot as plt
    def plot_true_vs_pred(y_true, y_pred, title_suffix):
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ['kPL', 'kVE', 'vB']
        plt.figure(figsize=(15, 4))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
            plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                    [y_true[:, i].min(), y_true[:, i].max()], 'r--')

            # Compute R²
            ss_res = np.sum((y_pred[:, i] - y_true[:, i]) ** 2)
            ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
            r2 = 1 - ss_res / ss_tot

            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(f"{labels[i]}: R²={r2:.3f}\n{title_suffix}")

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"true_vs_pred_{title_suffix}.png"))
        plt.close()
        import pandas as pd
        df = pd.DataFrame({
            'kPL_true': y_true[:, 0],
            'kVE_true': y_true[:, 1],
            'vB_true': y_true[:, 2],
            'kPL_pred': y_pred[:, 0],
            'kVE_pred': y_pred[:, 1],
            'vB_pred': y_pred[:, 2],
        })
        df.to_csv(os.path.join(run_dir, f"predictions_vs_truth_{title_suffix}.csv"), index=False)

    plot_true_vs_pred(y_true, y_pred, "Neural Network")

from fit_two_compartment import fit_traditional_2c_model
from two_compartment_generator import TwoCompartmentHPDataGenerator

# Traditional fit comparison
generator_instance = TwoCompartmentHPDataGenerator(time_points=time_points)
traditional_preds = []

X_test = X[X.shape[0] - len(Xn_test):]  # Align test indices

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
plot_true_vs_pred(y_true, y_pred_fit, "Traditional Fit")

# Step 12: Generate a summary report
print("\n--- Step 12: Generate Summary Report ---")
with open(os.path.join(run_dir, 'summary_report.md'), 'w') as f:
    f.write(f"# Hyperpolarized 13C MRI Analysis Demo Summary\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"## Dataset\n")
    f.write(f"- Total samples: {n_samples}\n")
    f.write(f"- Training samples: {Xr_train.shape[0]}\n")
    f.write(f"- Validation samples: {Xr_val.shape[0]}\n")
    f.write(f"- Test samples: {Xr_test.shape[0]}\n")
    f.write(f"- Time points: {n_timepoints} (TR=5s, 0-{(n_timepoints-1)*5}s)\n\n")
    
    f.write(f"## Parameter Ranges\n")
    f.write(f"- kPL: {generator.kpl_range[0]:.3f} - {generator.kpl_range[1]:.3f} s^-1\n")
    
    f.write(f"## Training\n")
    f.write(f"- Epochs: {n_epochs}\n")
    f.write(f"- Initial learning rate: 0.001\n")
    f.write(f"- Training time: {training_time:.2f} seconds\n")
    f.write(f"- Traditional fitting time: {tradfit_time:.2f} seconds")
    
    f.write(f"Test R² for kpl: {r2_kpl:.3f}")
    f.write(f"Test R² for kve: {r2_kve:.3f}")
    f.write(f"Test R² for vb:  {r2_vb:.3f}")



