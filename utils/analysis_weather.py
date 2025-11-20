import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats 

colors = {
    "temperature_2m": "#C4611A",
    "precipitation": "#3173EE",
    "wind_speed_10m": "#AD4DE0",
    "wind_gusts_10m": "#3C1053",
    "wind_direction_10m": "#075E50",
}


def dct_outliers(df: pd.DataFrame, target_col: str, freq_cutoff: float = 0.02, norm: str = 'ortho', type: int = 1, trim_percent: float = 0.05, n_std: float = 3.0 ) -> plt.Figure:
    series = df[[target_col]].copy()
    series = series.sort_index()
    series.index.name = "date"
    series[target_col] = (
        series[target_col]
        .interpolate(method="time")
        .bfill()
        .ffill()
    )
    signal = series[target_col].to_numpy()
    t = series.index


    # Prepare for DCT
    N = len(signal)
    W = np.arange(0,N)
    fourier_signal_coeffs = dct(signal, type=type, norm=norm)

    satv_coeffs = fourier_signal_coeffs.copy()
    # High-pass filtering: zero out low-frequency components
    cutoff_index = int(N * freq_cutoff)
    satv_coeffs[(W <= cutoff_index)] = 0

    # Inverse DCT to get SATV
    satv = idct(satv_coeffs, type=type, norm=norm)

    # Calculate robust center using trimmed mean
    robust_center = stats.trim_mean(satv, trim_percent)
    mad = stats.median_abs_deviation(satv)
    # Scale MAD to approximate the standard deviation of a normal distribution
    robust_std_estimate = mad * 1.4826 
    
    # Define SPC boundaries
    upper_bound = robust_center + n_std * robust_std_estimate
    lower_bound = robust_center - n_std * robust_std_estimate

    # We plot the boundaries on the main signal's mean for context on the original plot
    # I tried but somhow couldn't get it to plot correctly on the original signal
    trend = signal - satv
    ucl_curve = trend + upper_bound
    lcl_curve = trend + lower_bound

    outlier_mask = (satv > upper_bound) | (satv < lower_bound)
    outlier_points_x = series.loc[t[outlier_mask], target_col]

    line_color = colors[target_col]
    limit_color = "#726F6F"
    out_color = '#FF0000'

    fig, ax = plt.subplots()
    ax.plot(t, signal, color=line_color, label=f"{target_col.capitalize()} Data", linewidth=1.2)
    
    # Plot the UCL and LCL lines

    ax.plot(t, ucl_curve , color=limit_color, linestyle="--", linewidth=0.5, label=f"UCL (k={n_std})")
    ax.plot(t, lcl_curve, color=limit_color, linestyle="--", linewidth=0.5, label=f"LCL (k={n_std})")
    
    # Plot outliers using scatter if they exist
    if outlier_mask.any():
        # Plot using the correctly extracted outlier points
        ax.scatter(outlier_points_x.index, outlier_points_x.values, s=14, color=out_color, label="Outliers", zorder=3)

    ax.set_title(f"{target_col.capitalize()} with SPC Limits (robust, DCT high-pass)")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{target_col.capitalize()}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig



# --- LOF Time Series Anomaly Detection Function ---
def apply_lof_time_series(df: pd.DataFrame, target_col: str, n_neighbors: int = 20, contamination: float = 0.01) -> plt.Figure:
    
    # 1. Feature Engineering: Create lagged features
    X = df[[target_col]].sort_index().to_numpy().reshape(-1, 1)
    
    # The indices of the new data correspond to the original df indices after dropping initial NaNs
    original_indices = df.index

    # 2. Fit the LOF model on the multi-dimensional feature set
    # n_neighbors should generally be > window_size + 1
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X)
    outlier_mask = (y_pred == -1)

    # 3. Prepare data for plotting using the original time indices
    y = df.loc[original_indices, target_col].to_numpy() # Get corresponding original values
    idx_inliers = original_indices[~outlier_mask]
    idx_outliers = original_indices[outlier_mask]
    
    y_inliers = y[~outlier_mask]
    y_outliers = y[outlier_mask]

    # 4. Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(idx_inliers, y_inliers, label='Inlier', color='blue', s=15, alpha=0.6)
    ax.scatter(idx_outliers, y_outliers, label='Outlier', color='red', s=20, alpha=0.8)
    ax.set_title(f'{target_col.capitalize()} Outliers Detected by LOF (n_neighbors={n_neighbors}, contamination={contamination})')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{target_col.capitalize()}')
    ax.grid()
    ax.legend()

    return fig
