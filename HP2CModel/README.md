# Hyperpolarized 13C MRI Analysis - Two-Compartment Model (HP2CModel)

This project focuses on the simulation, training, and evaluation of models, particularly a hybrid neural network, for analyzing hyperpolarized 13C MRI data based on a **two-compartment exchange model**. This model typically involves vascular and extravascular pyruvate compartments, and an extravascular lactate compartment, allowing for the estimation of parameters like kPL (pyruvate-to-lactate conversion rate), kVE (vascular extravasation rate for pyruvate), and vB (fractional blood volume).

## Project Overview

The scripts in this folder are designed to:
* Generate synthetic data based on a two-compartment model with various complexities, including options for variable Arterial Input Functions (AIFs).
* Define and train a hybrid multi-head neural network designed to predict kinetic parameters (kPL, kVE, vB). This network uses both raw and normalized signal inputs.
* Perform inference using the trained model on synthetic data and NIFTI image data.
* Provide implementations of traditional model fitting approaches (e.g., using differential evolution) for comparison.

## File Descriptions

Here's a breakdown of the Python scripts included in the `HP2CModel` folder:

* **`two_compartment_generator.py`**:
    * **Purpose**: Generates synthetic hyperpolarized 13C MRI time-series data based on a detailed two-compartment model (vascular Pyruvate, extravascular Pyruvate, extravascular Lactate).
    * **Functionality**: Simulates the model incorporating parameters like kPL, kVE, vB, R1 relaxation rates, and crucially, a **variable gamma-variate Arterial Input Function (AIF)** with customizable parameters (t0, alpha, beta). It accounts for flip angle effects on apparent relaxation rates and can add noise to the signals.

* **`two_compartment_generator_vbonly.py`**:
    * **Purpose**: A simplified version of the two-compartment data generator.
    * **Functionality**: Generates data focusing on variations in kPL, kVE, and vB, using a fixed AIF shape and fixed R1 relaxation rates. The model solves for extravascular pyruvate (Pe) and lactate (Le), and combines them with a vascular pyruvate component (Pv, directly from AIF) to form the observable signals.

* **`hybrid_model_utils.py`**:
    * **Purpose**: Contains utility functions and the definition for the `HybridMultiHead` neural network model.
    * **Functionality**:
        * `HybridMultiHead` class: Defines a neural network with a shared base for normalized input and separate heads for kPL and kVE, plus a distinct head for vB using raw input.
        * Functions for loading NIFTI series (`load_nifti_series`).
        * Signal preprocessing/reshaping (`preprocess_signals`, `reshape_signals`).
        * Model evaluation (`evaluate_model`).
        * Plotting utilities for prediction scatter plots (`plot_prediction_scatter`, `plot_prediction_scatter2`) and weight histograms (`plot_weight_histograms`). The `plot_prediction_scatter2` function is specifically designed to handle and remove NaN values before plotting and calculating R² scores.


* **`train_hybrid_multihead.py`**:
    * **Purpose**: Trains the `HybridMultiHead` neural network model.
    * **Functionality**:
        * Generates a dataset using `TwoCompartmentHPDataGenerator` (specifically the `vbonly` version).
        * Prepares raw and normalized versions of the input signals.
        * Splits data into training, validation, and test sets.
        * Initializes and trains the `HybridMultiHead` model (imported from `hybrid_model_utils_positive`).
        * Saves the trained model weights.
        * Evaluates the trained NN on the test set and plots true vs. predicted values.
        * Performs a comparison by fitting a traditional two-compartment model (`fit_traditional_2c_model` from `fit_two_compartment_fixed.py`) to the test data and plots its predictions.
        * Generates a summary report of the training run.

* **`demo_generator_inference.py`**:
    * **Purpose**: Demonstrates generating synthetic data, loading a pre-trained `HybridMultiHead` model, and performing inference. It also compares the NN predictions with a traditional fitting method.
    * **Functionality**:
        * Generates a small synthetic dataset using `TwoCompartmentHPDataGenerator` (from `two_compartment_generator_vbonly.py`).
        * Loads a pre-trained `HybridMultiHead` model.
        * Performs inference on the generated data.
        * Plots prediction scatter plots for the neural network.
        * Fits the data using `fit_traditional_2c_model_de` (differential evolution based traditional fit) and plots its prediction scatter plots.

* **`demo_hybrid_nifti_inference.py`**:
    * **Purpose**: Applies a pre-trained `HybridMultiHead` model to 4D NIFTI data to generate 3D kinetic parameter maps.
    * **Functionality**:
        * Loads a pre-trained `HybridMultiHead` model.
        * Loads series of pyruvate and lactate NIFTI files (expects time points as separate files).
        * Performs optional Gaussian spatial smoothing.
        * Reshapes and preprocesses the NIFTI data (voxel-wise time series) into raw and normalized inputs for the model.
        * Performs inference using the model for each voxel.
        * Reshapes the predicted parameters (kPL, kVE, vB) back into 3D maps and saves them as NIFTI files.

* **`fit_two_compartment_de.py`**:
    * **Purpose**: Implements a traditional fitting approach for a simplified two-compartment model using differential evolution.
    * **Functionality**:
        * Defines a two-compartment ODE system (simpler than `two_compartment_generator.py`, directly models observable P and L without explicit vascular/extravascular separation for P in the ODEs themselves, though the signal model re-combines).
        * Simulates signals `S_pyr` and `S_lac` based on kPL, kVE, and vB.
        * Uses `scipy.optimize.differential_evolution` to find parameters that minimize the residual between observed and simulated signals.

* **`fit_two_compartment_fixed.py`**:
    * **Purpose**: Implements a traditional fitting approach for the detailed two-compartment model (matching `two_compartment_generator.py`) using `scipy.optimize.curve_fit`.
    * **Functionality**:
        * Defines the derivative for the 3-species ODE system (Pv, Pe, Le).
        * Uses a gamma-variate AIF (from `two_compartment_generator.py`).
        * Fits kPL, kVE, vB (and optionally R1p, R1l) by minimizing the difference between observed and simulated signals (S_pyr, S_lac).
        * Handles fixed or estimated R1 values.

* **`fit_two_compartment_fixed_de.py`**:
    * **Purpose**: Similar to `fit_two_compartment_fixed.py`, but uses `scipy.optimize.differential_evolution` for optimization instead of `curve_fit`.
    * **Functionality**: Adapts the model and objective function from `fit_two_compartment_fixed.py` for use with differential evolution.

* **`simulate_full_two_compartment.py`**:
    * **Purpose**: Simulates and fits a full two-compartment model, including a distinct vascular relaxation rate (R1v) and an explicit AIF infusion into the vascular compartment.
    * **Functionality**:
        * Defines a 3-species ODE: vascular pyruvate (P_v), extravascular pyruvate (P_e), and lactate (L).
        * Uses a gamma-variate AIF that is directly added as an input to the P_v compartment.
        * Simulates observable signals S_pyr and S_lac.
        * Provides a function `fit_two_compartment_de` to fit kPL, kVE, and vB using differential evolution based on these simulations.

## General Usage

1.  **Data Generation**:
    * `two_compartment_generator.py`: Can be run to understand the detailed data generation process with variable AIFs.
        ```bash
        python two_compartment_generator.py
        ```
    * `two_compartment_generator_vbonly.py`: Used by `train_hybrid_multihead.py` and `demo_generator_inference.py` for generating data with fixed AIFs and R1 rates.

2.  **Training the Hybrid Model**:
    * Execute `train_hybrid_multihead.py` to train the `HybridMultiHead` model.
        ```bash
        python train_hybrid_multihead.py
        ```
    * This script will generate data, train the model, save the model weights (e.g., to `output/demo_YYYYMMDD-HHMMSS/trained_hybrid_positive.pth` and `trained_hybrid_positive.pth`), and perform an initial evaluation against a traditional fitting method.

3.  **Inference on Synthetic Data**:
    * Run `demo_generator_inference.py` to see an example of loading the trained model and predicting on newly generated synthetic data.
    * You will need to update `weights_path` in the script to point to your trained model file (e.g., `output/demo_YYYYMMDD-HHMMSS/trained_hybrid_model.pth` or `trained_hybrid_positive.pth`).

4.  **Inference on NIFTI Data**:
    * Run `demo_hybrid_nifti_inference.py` to apply the trained model to 4D NIFTI images.
    * Update `weights_path` to your trained model file.
    * Place your pyruvate and lactate NIFTI time-series files (e.g., `pyr_t0.nii.gz`, `lac_t0.nii.gz`, etc.) in a `data/` subdirectory or update the file paths in `glob("data/pyr*.nii.gz")`.
    * The script will output 3D NIFTI maps for kPL, kVE, and vB.

5.  **Traditional Fitting Examples**:
    * `fit_two_compartment_de.py`: Example of fitting a simplified 2C model.
    * `fit_two_compartment_fixed.py`: Used by `train_hybrid_multihead.py` for comparison.
    * `fit_two_compartment_fixed_de.py`: Used by `demo_generator_inference.py` for comparison.
    * `simulate_full_two_compartment.py`: Demonstrates simulation and fitting of a more explicit 2C model with AIF infusion.

## Dependencies

Ensure you have the following Python libraries installed:
* numpy
* scipy
* torch (PyTorch)
* scikit-learn
* nibabel (for NIFTI processing)
* matplotlib
* pandas (used in `two_compartment_generator.py` and by `train_hybrid_multihead.py` for saving predictions)

You can typically install them using pip:
```bash
pip install numpy scipy torch torchvision torchaudio scikit-learn nibabel matplotlib pandas


Notes
1)The HybridMultiHead model leverages different normalizations of the input signal for different parameter heads, aiming to improve prediction accuracy.
2)The _vbonly version of the generator simplifies some aspects for more focused training or testing scenarios.
3)Pay attention to the weights_path variables in the demo/inference scripts and update them to point to your trained model files.
4)The NIFTI inference script assumes that your time points are stored as separate NIFTI files or can be loaded into a 4D array (X, Y, Z, T).
5)The hybrid_model_utils_positive.py file is critical for the train_hybrid_multihead.py script, ensure it's present and correctly defines the model, possibly with activation layers like ReLU or Sigmoid in the final layers of the heads to enforce positivity or specific ranges for the parameters.