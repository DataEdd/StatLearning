from typing import Union, Optional, Dict, Any
import pandas as pd
import numpy as np
import math
from scipy.stats import t


class SRegPred:
    def __init__(
        self,
        feature: Union[pd.Series, pd.DataFrame, np.ndarray],
        target: Union[pd.Series, pd.DataFrame, np.ndarray],
        handle_na: str = 'drop',
        keep_val: Optional[Dict[str, Any]] = None, 
        sample_train: float = 0.5,
        sample_test: float = 0.5
        ) -> None:
        
        self.X = self._to_1d_array(feature, name="feature")
        self.Y = self._to_1d_array(target, name="target")

        self.handle_na = handle_na.lower()
        # Handle keep_val default based on dtypes if handle_na == "keep"
        if self.handle_na == "keep":
            default_keep_val = {
                "feature": self._infer_default(self.X),
                "target": self._infer_default(self.Y),
            }
            # If keep_val is None, use default dict
            if keep_val is None:
                keep_val = default_keep_val
            else:
                if not isinstance(keep_val, dict):
                    raise TypeError("`keep_val` must be a dict like {'feature': 0, 'target': 'missing'}")

                for key in keep_val:
                    if key not in ("feature", "target"):
                        raise ValueError(f"`keep_val` can only contain keys 'feature' and/or 'target', not '{key}'")

                for key in ("feature", "target"):
                    if key not in keep_val:
                        keep_val[key] = default_keep_val[key]

                for k, v in keep_val.items():
                    if not isinstance(v, (int, float, str)):
                        raise TypeError(
                            f"`keep_val['{k}']` must be int, float, or str. Got {type(v).__name__}"
                        )

            self.keep_val = keep_val
        else:
            self.keep_val = None

        self.combined_na = None
        self.checks()

        # Adjust sample ratios dynamically
        if sample_train != 0.5 and sample_test == 0.5:
            sample_test = 1 - sample_train
        elif sample_test != 0.5 and sample_train == 0.5:
            sample_train = 1 - sample_test
        elif (sample_train != 0.5 or sample_test != 0.5) and not np.isclose(sample_train + sample_test, 1.0):
            raise ValueError("sample_train and sample_test must sum to 1.")

        self.sample_train = sample_train
        self.sample_test = sample_test
        
        self.split()
        self.X_train_bar = np.nanmean(self.X_train)
        self.X_test_bar = np.nanmean(self.X_test)
        self.Y_train_bar = np.nanmean(self.Y_train)
        self.Y_test_bar = np.nanmean(self.Y_test)

        self.predict()
        self.values()


    def _to_1d_array(self, data, name="data") -> np.ndarray:
        '''Convert feature and target to 1D numpy arrays. assumes single column for DataFrame and columns filled with same data types.''' 

        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError(f"{name} DataFrame must have exactly one column")
            data = data.iloc[:, 0]
        if isinstance(data, pd.Series):
            return data.to_numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"{name} must be a pandas Series, DataFrame with one column, or NumPy array")

    def _infer_default(self, arr: np.ndarray):
        if np.issubdtype(arr.dtype, np.number):
            return 0
        elif np.issubdtype(arr.dtype, np.str_) or arr.dtype == object:
            return "missing"
        else:
            raise TypeError(f"Unsupported dtype {arr.dtype} for missing value default. Please provide keep_val explicitly.")

    def checks(self):
        '''
        Checks for missing values and handles them according to self.handle_na
        Also checks for categorical data
        '''
        # Combine X and Y into a single array for joint NA detection
        combined = np.column_stack((self.X, self.Y))

        # Mask of rows where either column has missing values
        na_mask = pd.isna(combined).any(axis=1)

        # Save affected rows for Nan analysis later
        self.combined_na = combined[na_mask]

        # Then apply NA handling strategy
        match self.handle_na:
            case "drop":
                combined = combined[~na_mask]
                self.X, self.Y = combined[:, 0], combined[:, 1]

            case "keep":
                self.X = np.where(pd.isna(self.X), self.keep_val["feature"], self.X) # type: ignore
                self.Y = np.where(pd.isna(self.Y), self.keep_val["target"], self.Y) # type: ignore

            case "mean":
                if not np.issubdtype(self.X.dtype, np.number) or not np.issubdtype(self.Y.dtype, np.number):
                    raise TypeError("Cannot compute mean on non-numeric data.")
                x_mean = np.nanmean(self.X)
                y_mean = np.nanmean(self.Y)
                self.X = np.where(pd.isna(self.X), x_mean, self.X)
                self.Y = np.where(pd.isna(self.Y), y_mean, self.Y)

            case "median":
                if not np.issubdtype(self.X.dtype, np.number) or not np.issubdtype(self.Y.dtype, np.number):
                    raise TypeError("Cannot compute median on non-numeric data.")
                x_median = np.nanmedian(self.X)
                y_median = np.nanmedian(self.Y)
                self.X = np.where(pd.isna(self.X), x_median, self.X)
                self.Y = np.where(pd.isna(self.Y), y_median, self.Y)

            case "error":
                if na_mask.any():
                    raise ValueError("Missing values detected in input data.")

            case _:
                raise ValueError(f"Unsupported handle_na method: {self.handle_na}")
    
    def split(self):
        combined = np.column_stack((self.X,self.Y))
        rng = np.random.default_rng()
        rng.shuffle(combined, axis=0)

        split_idx = int(len(combined) * self.sample_train)

        train_data = combined[:split_idx]
        test_data = combined[split_idx:]

        self.X_train, self.Y_train = train_data[:, 0], train_data[:, 1]
        self.X_test, self.Y_test = test_data[:, 0], test_data[:, 1]

        return self.X_train, self.X_test, self.Y_train, self.Y_test
    

    def predict(self):
        '''
        Calculates beta1 and beta0 by minimizing RSS
        '''
        num = np.sum((self.X_train - self.X_train_bar) * (self.Y_train - self.Y_train_bar))
        den = np.sum((self.X_train - self.X_train_bar) ** 2)

        if den == 0:
            raise ValueError("Cannot compute slope: zero variance in X_train")

        self.beta1_est = num / den
        self.beta1_est_sign = "-" if self.beta1_est < 0 else "+"
        self.beta0_est = self.Y_train_bar - self.beta1_est * self.X_train_bar
        
        # Predictions on training set
        self.y_pred_train = self.beta0_est + self.beta1_est * self.X_train

        # Predictions on test set
        self.y_pred_test = self.beta0_est + self.beta1_est * self.X_test

        return f"(Y = {self.beta0_est} {self.beta1_est_sign} {abs(self.beta1_est)} X)"
    
    def values(self):
        '''
        Calculates residuals, RSS, and variance of residuals (sigma^2)
        Assumes errors for each observation have common variance and are uncorrelated
        '''
        self.residuals_train = self.Y_train - self.y_pred_train # Residuals on training set
        self.residuals_test = self.Y_test - self.y_pred_test # Residuals on test set

        self.RSS_train = np.sum(self.residuals_train ** 2) # RSS (Residual Sum of Squares) of training set
        self.RSS_test = np.sum(self.residuals_test ** 2) # RSS (Residual Sum of Squares) of test set
        
        self.n_train = len(self.Y_train) # Number of training samples
        self.df = self.n_train - 2 # Degree of Freedom for LR
        
        if self.n_train<= 2: # Degrees of freedom (n - 2) for simple linear regression
            raise ValueError("Not enough data points to compute variance of residuals")
        
        self.sigma_squared = self.RSS_train / (self.n_train - 2) # Variance of residuals (sigma^2)
        self.RSE = math.sqrt(self.sigma_squared) # Residual Standard Error. Measure of the lack of fit of the mode to the data

        Sxx = np.sum((self.X_train - self.X_train_bar) ** 2) # Sxx = sum_i^n (x_i - x_bar)^2

        self.SE_beta1_squared = self.sigma_squared / Sxx # Variance of the estimated regression slope
        self.SE_beta1 = np.sqrt(self.SE_beta1_squared) # Standard Error of the estimated regression slope

        self.SE_beta0_squared = self.sigma_squared * (1 / self.n_train + (self.X_train_bar ** 2) / Sxx) # Variance of the estimated regression intercept
        self.SE_beta0 = np.sqrt(self.SE_beta0_squared) # Standard Error of the estimated regression intercept

        self.t_beta1 = self.beta1_est / self.SE_beta1 # t-statistic for estimated regression slope
        self.p_beta1 = t.sf(abs(self.t_beta1), self.df) * 2 # p-value for estimated regression slope
        
        self.t_beta0= self.beta0_est / self.SE_beta0 # t-statistic for estimated regression intercept
        self.p_beta0 = t.sf(abs(self.t_beta0), self.df) * 2 # p-value for estimated regression intercept
        
        self.TSS_train = np.sum((self.Y_train - np.mean(self.Y_train)) ** 2) # TSS (Total Sum of Squares) of training set

        self.Rsq = 1 - self.RSS_train/self.TSS_train # R^2 Statistic. Measures the proportion of variability in Y that can be explained using X

    def Confidence(self, value, alpha):
        return
    






