***

# STATISTICALLEARNING

A Python package for simple and multiple linear regression with robust missing value handling, data splitting, result diagnostics, and extensible interfaces for future statistical features.

***

## Table of Contents

- [Installation \& Environment](#installation--environment)
- [Usage Guide](#usage-guide)
- [Package Overview](#package-overview)
- [API and Module Documentation](#api-and-module-documentation)
    - [SimpleLinearRegression (`predict.py`)](#simplelinearregression-predictpy)
        - [Current Capabilities](#current-capabilities)
        - [Planned Methods](#planned-methods)
- [Directory Structure](#directory-structure)
- [License](#license)

***

## Installation \& Environment

1. **Clone this repository:**

```bash
git clone <repo-url>
cd STATISTICALLEARNING
```

2. **Create a virtual environment (recommended):**

```bash
python3 -m venv myenv
source myenv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```


***

## Usage Guide

1. **Activate your environment if not already:**

```bash
source myenv/bin/activate
```

2. **Run scripts:**
    - Example:

```bash
python SimpleLinearRegression/predict.py
```

    - Or open and execute cells in `testing.ipynb` for interactive testing.
3. **Add your data in the `testdata/` directory as required.**

***

## Package Overview

This repository is organized for modular statistical learning. Each algorithm or method is in its own folder.

- `SimpleLinearRegression/`: Single-feature OLS regression
- `MultipleLinearRegression/`: (Planned/Under construction)
- `LinearRegression/`: (Base or shared code)

There are supporting utility scripts, sample Jupyter notebooks, and clear division for unit testing, dependencies, and documentation.

***

## API and Module Documentation

### `SimpleLinearRegression/predict.py`

#### Class: `SRegPred`

Performs **simple linear regression** with comprehensive missing value strategies and train/test splitting.

**Constructor**

```python
SRegPred(
    feature: pd.Series | pd.DataFrame | np.ndarray,
    target: pd.Series | pd.DataFrame | np.ndarray,
    handle_na: str = "drop",  # 'drop', 'mean', 'median', 'keep', 'error'
    keep_val: dict | None = None,
    sample_train: float = 0.5,
    sample_test: float = 0.5
)
```


#### Current Capabilities

- **_to_1d_array**: Ensures input is a 1D NumPy array from pandas/numpy inputs.
- **_infer_default**: Chooses a sensible fill value (`0` for numeric, `'missing'` for string) for missing data if `handle_na='keep'`.
- **checks**: Handles missing values according to `handle_na` parameter:
    - `'drop'`: Drops rows with missing in either feature/target.
    - `'mean'/'median'`: Fills with mean/median (only for numeric).
    - `'keep'`: Fills with specified constant by dtype.
    - `'error'`: Raises error if any missing value present.
- **split**: Random train/test split (default 50/50, configurable).
- **predict**: Fits and stores regression coefficients (`beta0`, `beta1`), generates predictions for both train and test sets.
- **values**: Computes residuals, RSS, RSE, estimated variances, t-stats, p-values for slope/intercept, TSS, and R² (goodness of fit).
- **Handles both numerical and categorical fill strategies for missing data.**
- **Diagnostic values:** (standard errors, t, p, RSE, R², etc).


#### Planned Methods

- **summarize**: Print or export model summary (coefficients, diagnostics, p-values, fit, etc).
- **plots**: Visualizations for regression line, residuals, diagnostics.
- **confidence**: Calculate confidence intervals for coefficients or predictions.
- **predict_new**: Predict for new data, with intervals.
- **save/load**: Serialize model and predictions to/from file.

*(Stubs and interface for these planned features will be added in future updates as per class notes.)*

***

## Directory Structure

```
STATISTICALLEARNING/
│
├── LinearRegression/                # Shared or base code (planned)
├── MultipleLinearRegression/        # For multi-feature regression (planned)
└── SimpleLinearRegression/
    ├── __pycache__/
    ├── testdata/                    # Place test datasets here
    ├── main.py                      # Example usage and CLI entry
    ├── predict.py                   # Core simple linear regression implementation
    └── testing.ipynb                # Sample interactive notebook
│
├── myenv/                           # Python virtual environment (not versioned)
├── README.TXT                       # (You are reading this!)
└── requirements.txt                 # Dependencies for pip
```


***

## License

[Your License Here]

***

### Notes

- For any "method not implemented" error, see the 'Planned Methods' section above.
- For bug reports, usage questions, or contributions: please open an issue or PR.

***

*This README is current as of code structure and function signatures as seen on Aug 14, 2025.*

***

Let me know if you want specific docstrings, sample usage, or more details on future features!

<div style="text-align: center">⁂</div>

[^1]: Screenshot-2025-08-14-at-16.35.39.jpg

