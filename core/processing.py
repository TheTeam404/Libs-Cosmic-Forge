# -*- coding: utf-8 -*-
"""
Core signal processing functions for LIBS spectra.

Includes baseline correction, smoothing, peak profile functions,
and noise analysis capabilities.
"""

import logging
import math # Needed for log in threshold calculation
import numpy as np
from typing import Tuple, Optional, List, Union

# --- SciPy Import Handling ---
# Check for SciPy availability, needed for Savitzky-Golay smoothing.
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning(
        "SciPy not found. Savitzky-Golay smoothing will be unavailable. "
        "Install with 'pip install scipy'."
    )
    # Define a placeholder function to raise an error if called
    def savgol_filter(*args, **kwargs):
        raise ImportError(
            "SciPy is required for Savitzky-Golay smoothing but is not installed."
        )

# --- PyWavelets Import Handling ---
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
    logging.debug("PyWavelets library found. Wavelet smoothing enabled.")
except ImportError:
    PYWAVELETS_AVAILABLE = False
    logging.warning(
        "PyWavelets not found. Wavelet smoothing will be unavailable. "
        "Install with 'pip install PyWavelets'."
    )
    # Define a placeholder function to raise an error if called
    def smooth_wavelet(*args, **kwargs):
        raise ImportError(
            "PyWavelets is required for wavelet smoothing but is not installed."
        )


# --- Utility Import ---
try:
    from utils.helpers import ensure_odd
except ImportError:
    logging.warning("utils.helpers.ensure_odd not found. Using basic implementation.")
    def ensure_odd(n: int) -> int:
        """Ensures an integer is odd."""
        n = int(n)
        return n if n % 2 != 0 else n + 1

# --- Constants ---
# Factor to convert Gaussian sigma to FWHM
FWHM_GAUSS_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0)) # Approx 2.35482
# Small epsilon to avoid division by zero or log(0) issues
EPSILON = 1e-9

# --- Profile Functions ---
# (gaussian, lorentzian, pseudo_voigt functions remain the same as before)
def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Calculates a Gaussian profile."""
    if sigma <= EPSILON: logging.warning(f"Gaussian sigma ({sigma:.2e}) invalid."); return np.full_like(x, np.inf)
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))

def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    """Calculates a Lorentzian profile."""
    if gamma <= EPSILON: logging.warning(f"Lorentzian gamma ({gamma:.2e}) invalid."); return np.full_like(x, np.inf)
    return amplitude * (gamma**2 / ((x - center)**2 + gamma**2))

def pseudo_voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, eta: float) -> np.ndarray:
    """Calculates a Pseudo-Voigt profile."""
    if sigma <= EPSILON: logging.warning(f"Pseudo-Voigt sigma ({sigma:.2e}) invalid."); return np.full_like(x, np.inf)
    eta_bounded = np.clip(eta, 0.0, 1.0); gamma = max(EPSILON, (sigma * FWHM_GAUSS_FACTOR) / 2.0)
    gauss_part = (1.0 - eta_bounded) * np.exp(-((x - center)**2) / (2 * sigma**2))
    loren_part = eta_bounded * (gamma**2 / ((x - center)**2 + gamma**2))
    return amplitude * (gauss_part + loren_part)

# --- Baseline Correction Algorithms ---
# (_interpolate_finite, baseline_poly, baseline_snip remain the same as before)
def _interpolate_finite(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Internal helper to interpolate NaN/Inf values using linear interpolation."""
    y_interp = y.copy()
    finite_mask = np.isfinite(y_interp)
    all_finite = np.all(finite_mask)
    if not all_finite:
        nan_mask = ~finite_mask; finite_indices = np.flatnonzero(finite_mask); nan_indices = np.flatnonzero(nan_mask)
        if len(finite_indices) < 2: logging.error("Cannot interpolate NaNs: < 2 finite points."); y_interp[nan_mask] = 0.0; finite_mask = np.isfinite(y_interp); return y_interp, finite_mask, np.any(finite_mask)
        try: y_interp[nan_mask] = np.interp(nan_indices, finite_indices, y_interp[finite_indices]); logging.warning(f"Interpolated {np.sum(nan_mask)} NaN/Inf values.")
        except Exception as e: logging.error(f"NaN interpolation failed: {e}. Setting NaNs to 0.", exc_info=True); y_interp[~np.isfinite(y_interp)] = 0.0; finite_mask = np.isfinite(y_interp); return y_interp, finite_mask, np.any(finite_mask)
        if not np.all(np.isfinite(y_interp)): logging.error("Interpolation failed to remove all NaNs/Infs."); finite_mask = np.isfinite(y_interp)
    return y_interp, finite_mask, True

def baseline_poly(wavelengths: np.ndarray, intensity: np.ndarray, order: int = 3, percentile: float = 10.0, max_iterations: int = 10, tolerance: float = 0.001, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ Estimates baseline using iterative polynomial fitting on points below a percentile. """
    # (Implementation remains the same as before)
    if order < 0: logging.debug("Poly baseline skipped (order < 0)."); return intensity, np.zeros_like(intensity)
    if not (isinstance(wavelengths, np.ndarray) and isinstance(intensity, np.ndarray)): raise TypeError("Wavelengths/intensity must be NumPy arrays.")
    if wavelengths.shape != intensity.shape: raise ValueError(f"Shape mismatch: Wavelength ({wavelengths.shape}) vs Intensity ({intensity.shape})")
    n_points = len(wavelengths); min_points_required = order + 1
    if n_points < min_points_required: logging.warning(f"Poly baseline skipped: Need >= {min_points_required} pts for order {order}, got {n_points}."); return intensity, np.zeros_like(intensity)
    if not (0 < percentile <= 100): logging.warning(f"Invalid percentile ({percentile}). Using default: 10%."); percentile = 10.0
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok: logging.error("NaN handling failed. Skipping poly baseline."); return intensity, np.zeros_like(intensity)
    current_intensity = intensity_processed; baseline = np.zeros_like(intensity_processed); iteration = 0
    for iteration in range(max_iterations if max_iterations > 0 else 1):
        try:
            valid_for_percentile = intensity_processed[original_finite_mask] if iteration == 0 else current_intensity[original_finite_mask]
            if len(valid_for_percentile) == 0: logging.error("No finite points for percentile. Skipping."); return intensity, np.zeros_like(intensity)
            threshold = np.percentile(valid_for_percentile, percentile)
            mask = (current_intensity <= threshold) & original_finite_mask; n_masked_pts = np.sum(mask)
            if n_masked_pts < min_points_required:
                logging.warning(f"Iter {iteration+1}: Only {n_masked_pts} pts below {percentile:.1f}% threshold. Using fallback."); finite_indices = np.where(original_finite_mask)[0]
                if len(finite_indices) < min_points_required: logging.error(f"Fallback failed: Only {len(finite_indices)} finite points. Skipping."); return intensity, np.zeros_like(intensity)
                sorted_finite_indices = finite_indices[np.argsort(intensity_processed[finite_indices])]; fallback_indices = sorted_finite_indices[:min_points_required]
                mask = np.zeros_like(intensity_processed, dtype=bool); mask[fallback_indices] = True; n_masked_pts = np.sum(mask)
            x_masked, y_masked = wavelengths[mask], intensity_processed[mask]
            if not (np.all(np.isfinite(x_masked)) and np.all(np.isfinite(y_masked))) or len(x_masked) < min_points_required: logging.error("Non-finite/insufficient points for polyfit. Skipping."); return intensity, np.zeros_like(intensity)
            coeffs = np.polyfit(x_masked, y_masked, order); new_baseline = np.polyval(coeffs, wavelengths)
            if max_iterations > 1: diff = np.abs(baseline - new_baseline);
            if np.all(diff[original_finite_mask] < tolerance): logging.info(f"Poly baseline converged after {iteration + 1} iterations."); baseline = new_baseline; break
            baseline = new_baseline; current_intensity = intensity_processed - baseline
        except (np.linalg.LinAlgError, ValueError) as e: logging.error(f"Poly baseline fitting failed: {e}", exc_info=True); return intensity, np.zeros_like(intensity)
        except Exception as e: logging.error(f"Unexpected error in poly baseline: {e}", exc_info=True); return intensity, np.zeros_like(intensity)
    corrected_intensity = intensity_processed - baseline; corrected_intensity[~original_finite_mask] = intensity[~original_finite_mask]; baseline[~original_finite_mask] = np.nan
    logging.info(f"Poly baseline applied (order={order}, iter={iteration+1})."); return corrected_intensity, baseline

def baseline_snip(wavelengths: np.ndarray, intensity: np.ndarray, max_iterations: int = 100, increasing_window: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ Estimates baseline using the SNIP algorithm. """
    # (Implementation remains the same as before)
    if not isinstance(intensity, np.ndarray): raise TypeError("Intensity must be NumPy array.")
    n_points = len(intensity);
    if n_points < 3: logging.warning(f"SNIP requires >= 3 points. Skipping."); return intensity, np.zeros_like(intensity)
    if max_iterations < 1: logging.warning(f"SNIP max_iterations < 1. Setting to 1."); max_iterations = 1
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok: logging.error("NaN handling failed. Skipping SNIP."); return intensity, np.zeros_like(intensity)
    baseline = intensity_processed.copy(); logging.info(f"Starting SNIP (max_iter={max_iterations}, increasing={increasing_window}).")
    try:
        for k in range(1, max_iterations + 1):
            window_size = k if increasing_window else max_iterations
            if window_size >= n_points // 2: logging.debug(f"SNIP iter {k}: Window size limit reached."); break
            prev_baseline = baseline.copy(); avg_neighbors = (prev_baseline[:-2 * window_size] + prev_baseline[2 * window_size:]) / 2.0
            points_to_clip = prev_baseline[window_size:-window_size] > avg_neighbors; baseline[window_size:-window_size][points_to_clip] = avg_neighbors[points_to_clip]
        baseline = np.minimum(baseline, intensity_processed); corrected_intensity = intensity_processed - baseline
        corrected_intensity[~original_finite_mask] = intensity[~original_finite_mask]; baseline[~original_finite_mask] = np.nan
        logging.info("SNIP baseline complete."); return corrected_intensity, baseline
    except Exception as e: logging.error(f"Error during SNIP: {e}", exc_info=True); return intensity, np.zeros_like(intensity)

# --- Smoothing Algorithms ---

def smooth_savitzky_golay(intensity: np.ndarray, smoothing_window: int = 11, smoothing_polyorder: int = 3, **kwargs) -> np.ndarray:
    """ Applies Savitzky-Golay smoothing filter. """
    # (Implementation remains the same as before)
    if not SCIPY_AVAILABLE: logging.error("Cannot smooth: SciPy unavailable."); return intensity
    if not isinstance(intensity, np.ndarray): raise TypeError("Intensity must be NumPy array.")
    n_points = len(intensity);
    if n_points < 3: logging.warning(f"Data too short ({n_points}) for Sav-Gol. Returning original."); return intensity
    try:
        wl = ensure_odd(int(smoothing_window));
        if wl < 3: logging.warning(f"Sav-Gol window < 3. Adjusting to 3."); wl = 3
        po = int(smoothing_polyorder);
        if po < 0: logging.warning(f"Sav-Gol polyorder < 0. Adjusting to 0."); po = 0
        if wl > n_points: old_wl = wl; wl = n_points if n_points % 2 != 0 else n_points - 1; wl = max(3, wl); logging.warning(f"Sav-Gol window ({old_wl}) > data length ({n_points}). Adjusting to {wl}.")
        if po >= wl: old_po = po; po = max(0, wl - 2); logging.warning(f"Sav-Gol polyorder ({old_po}) >= window ({wl}). Adjusting to {po}.")
    except (ValueError, TypeError) as e: logging.error(f"Invalid Sav-Gol parameters: {e}. Returning original."); return intensity
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok: logging.error("NaN handling failed before smoothing. Returning original."); return intensity
    elif not np.all(original_finite_mask): logging.debug("Interpolated NaNs/Infs before smoothing.")
    try:
        if wl > len(intensity_processed) or po >= wl: logging.error(f"Internal Error: Adjusted Sav-Gol params invalid. Skipping."); return intensity
        smoothed_intensity = savgol_filter(intensity_processed, window_length=wl, polyorder=po)
        smoothed_intensity[~original_finite_mask] = intensity[~original_finite_mask]
        logging.info(f"Applied Savitzky-Golay smoothing (window={wl}, order={po})."); return smoothed_intensity
    except Exception as e: logging.error(f"Error applying Sav-Gol: {e}. Returning original.", exc_info=True); return intensity

# ***** NEW FUNCTION ADDED *****
def smooth_wavelet(
    intensity: np.ndarray,
    wavelet: str = 'sym4',
    level: Optional[int] = None,
    mode: str = 'soft', # 'soft' or 'hard'
    threshold_method: str = 'VisuShrink', # 'VisuShrink' or 'Manual' (or others if implemented)
    threshold_value: Optional[float] = None, # Only used if threshold_method='Manual'
    sigma: Optional[float] = None, # Noise std dev, estimate if None
    wavelet_mode: str = 'symmetric' # PyWavelets signal extension mode
) -> np.ndarray:
    """
    Applies wavelet denoising to the intensity data.

    Requires the PyWavelets library to be installed (`pip install PyWavelets`).

    Args:
        intensity: Intensity array to denoise.
        wavelet: Name of the wavelet to use (e.g., 'db4', 'sym4', 'coif2').
        level: Decomposition level. If None, it's estimated automatically.
        mode: Thresholding mode ('soft' or 'hard').
        threshold_method: Method to determine threshold ('VisuShrink', 'Manual').
                          'BayesShrink', 'SureShrink' are possibilities but require more complex implementation or external libraries.
        threshold_value: Explicit threshold value (only used if threshold_method='Manual').
        sigma: Estimated noise standard deviation. If None, it's estimated using
               Median Absolute Deviation (MAD) on the finest detail coefficients.
        wavelet_mode: Signal extension mode used by PyWavelets (e.g., 'symmetric', 'zero').

    Returns:
        The denoised intensity array. Returns the original array if PyWavelets
        is unavailable or denoising fails.
    """
    if not PYWAVELETS_AVAILABLE:
        logging.error("Cannot apply wavelet smoothing: PyWavelets library not found.")
        return intensity
    if not isinstance(intensity, np.ndarray):
        raise TypeError("Intensity must be a NumPy array.")

    n_points = len(intensity)
    if n_points < 4: # Wavelets need some data
        logging.warning(f"Data too short ({n_points}) for wavelet smoothing. Returning original.")
        return intensity

    # --- Handle NaNs/Infs ---
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok:
         logging.error("Failed to handle NaNs/Infs prior to wavelet smoothing. Returning original.")
         return intensity
    elif not np.all(original_finite_mask):
         logging.debug("Applied linear interpolation to handle NaNs/Infs before wavelet smoothing.")

    try:
        # --- Determine Decomposition Level ---
        if level is None or level <= 0:
            try:
                # Estimate max level based on data length and wavelet
                # Note: dwtn_max_level requires wavelet object, max_level is simpler
                # Use simple log2 approach as a robust default, maybe adjust later
                level = math.floor(math.log2(n_points)) - 2 # Heuristic, can be adjusted
                level = max(1, level) # Ensure at least level 1
                logging.debug(f"Auto-determined wavelet decomposition level: {level}")
            except Exception as lvl_e:
                 logging.warning(f"Could not auto-determine wavelet level: {lvl_e}. Defaulting to 4.")
                 level = 4 # Fallback level

        # --- Decompose Signal ---
        # Perform multilevel Discrete Wavelet Transform (DWT)
        coeffs = pywt.wavedec(intensity_processed, wavelet, level=level, mode=wavelet_mode)
        # coeffs is a list: [cAn, cDn, cDn-1, ..., cD1] (Approx, Detail_n, ..., Detail_1)

        # --- Estimate Noise Standard Deviation (Sigma) ---
        sigma_est = sigma
        if sigma_est is None:
            try:
                # Estimate sigma from the highest frequency detail coefficients (cD1) using MAD
                detail_coeffs_level1 = coeffs[-1]
                sigma_est = pywt.mad(detail_coeffs_level1)
                if sigma_est is None or sigma_est < EPSILON:
                    logging.warning(f"MAD noise estimate failed or was near zero ({sigma_est}). Using small default.")
                    sigma_est = 1e-6 # Fallback small sigma
                else:
                     logging.debug(f"Estimated noise sigma (MAD): {sigma_est:.4g}")
            except Exception as sigma_e:
                logging.error(f"Error estimating noise sigma using MAD: {sigma_e}. Using small default.", exc_info=True)
                sigma_est = 1e-6 # Fallback

        # --- Calculate Threshold Value ---
        thresh_value = threshold_value # Use manual value if provided and method is Manual
        if threshold_method.lower() == 'visushrink':
             # Universal Threshold (VisuShrink)
             thresh_value = sigma_est * np.sqrt(2 * np.log(n_points))
             logging.debug(f"Using VisuShrink threshold value: {thresh_value:.4g}")
        elif threshold_method.lower() == 'manual':
            if thresh_value is None:
                logging.error("Threshold method is 'Manual' but no threshold_value provided. Denoising will likely fail.")
                # Could default to VisuShrink here or raise error
                # Let's default to VisuShrink threshold if manual value is missing
                thresh_value = sigma_est * np.sqrt(2 * np.log(n_points))
                logging.warning(f"Manual threshold not set, using calculated VisuShrink value: {thresh_value:.4g}")
        # Add elif for 'BayesShrink', 'SureShrink' etc. if implemented later
        else:
            logging.warning(f"Unsupported threshold_method '{threshold_method}'. Using VisuShrink.")
            thresh_value = sigma_est * np.sqrt(2 * np.log(n_points))

        # --- Apply Thresholding ---
        # Threshold ONLY the detail coefficients (coeffs[1] to coeffs[level])
        coeffs_thresh = [coeffs[0]] # Keep approximation coefficients untouched
        for i in range(1, len(coeffs)):
            coeffs_thresh.append(pywt.threshold(coeffs[i], value=thresh_value, mode=mode))

        # --- Reconstruct Signal ---
        denoised_intensity = pywt.waverec(coeffs_thresh, wavelet, mode=wavelet_mode)

        # --- Finalize ---
        # Ensure output length matches input (sometimes reconstruction adds/removes points)
        if len(denoised_intensity) != n_points:
             logging.warning(f"Wavelet reconstruction length ({len(denoised_intensity)}) differs from original ({n_points}). Adjusting.")
             # Simple truncation/padding - might need refinement
             if len(denoised_intensity) > n_points:
                 denoised_intensity = denoised_intensity[:n_points]
             else: # Pad if too short (less common)
                 pad_width = n_points - len(denoised_intensity)
                 # Use edge padding for potentially better results than zero padding
                 denoised_intensity = np.pad(denoised_intensity, (0, pad_width), mode='edge')


        # Restore original NaNs/Infs
        denoised_intensity[~original_finite_mask] = intensity[~original_finite_mask]

        logging.info(f"Applied Wavelet denoising (wavelet='{wavelet}', level={level}, mode='{mode}', thresh='{threshold_method}').")
        return denoised_intensity

    except Exception as e:
        logging.error(f"Error during Wavelet smoothing: {e}", exc_info=True)
        return intensity # Return original data on failure

# ***** END OF NEW FUNCTION *****

# --- Noise Analysis ---
# (analyze_noise function remains the same as before)
def analyze_noise(wavelengths: np.ndarray, intensity: np.ndarray, signal_free_regions: Optional[List[Tuple[float, float]]] = None) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]]]:
    """ Estimates noise level from signal-free regions of the spectrum. """
    # (Implementation remains the same as before)
    if not (isinstance(wavelengths, np.ndarray) and isinstance(intensity, np.ndarray)): raise TypeError("Wavelengths/intensity must be NumPy arrays.")
    if wavelengths.shape != intensity.shape: raise ValueError("Wavelength/Intensity shape mismatch.")
    if signal_free_regions is None or not signal_free_regions: logging.warning("No signal-free regions for noise analysis."); return None, signal_free_regions
    noise_segments = []; valid_regions_used = []
    for region in signal_free_regions:
        try:
            start_wl, end_wl = map(float, region);
            if start_wl >= end_wl: logging.warning(f"Skipping invalid noise region {region}."); continue
            region_mask = (wavelengths >= start_wl) & (wavelengths <= end_wl); intensity_in_region = intensity[region_mask]
            if intensity_in_region.size == 0: logging.warning(f"Skipping noise region {region}: No data points."); continue
            finite_intensity = intensity_in_region[np.isfinite(intensity_in_region)]
            if finite_intensity.size < 2: logging.warning(f"Skipping noise region {region}: < 2 finite points."); continue
            noise_segments.append(finite_intensity); valid_regions_used.append(region)
        except (TypeError, ValueError) as e: logging.warning(f"Skipping invalid noise region {region}: {e}.")
        except Exception as e: logging.error(f"Error processing noise region {region}: {e}", exc_info=True)
    if not noise_segments: logging.error("No valid data points in specified signal-free regions."); return None, valid_regions_used
    try:
        all_noise_points = np.concatenate(noise_segments)
        if all_noise_points.size < 2: logging.error("Insufficient total points (< 2) for noise estimate."); return None, valid_regions_used
        noise_std_dev = np.std(all_noise_points); logging.info(f"Estimated noise std dev: {noise_std_dev:.4f} (from {len(valid_regions_used)} regions, {all_noise_points.size} points)."); return noise_std_dev, valid_regions_used
    except Exception as e: logging.error(f"Failed to calculate noise std dev: {e}", exc_info=True); return None, valid_regions_used


# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    # (Example usage code can remain largely the same, but add a test for smooth_wavelet)
    # ... [previous example code for generating data] ...

    print("\n--- Testing Wavelet Smoothing ---")
    if PYWAVELETS_AVAILABLE:
        smoothed_wavelet = smooth_wavelet(
            intensity_with_nans, # Use data with NaNs to test handling
            wavelet='sym4',
            level=None, # Auto level
            mode='soft',
            threshold_method='VisuShrink' # Using simplest automatic threshold
        )
        print(f"Wavelet smoothed mean: {np.nanmean(smoothed_wavelet):.2f}")

        # Plotting (inside the existing try-except block)
        try:
            import matplotlib.pyplot as plt
            # ... [Previous plot setup and subplots 1 & 2] ...
            plt.subplot(3, 1, 1)
            plt.plot(wavelengths, intensity_raw, label='Raw Intensity', alpha=0.7)
            plt.plot(wavelengths, baseline_true, 'k--', label='True Baseline')
            plt.plot(wavelengths, baseline_poly_calc, 'r-', label='Poly Baseline (O=2, P=5)')
            plt.plot(wavelengths, baseline_snip_calc, 'g-', label='SNIP Baseline (Iter=50)')
            plt.title('Raw Data and Estimated Baselines')
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(wavelengths, intensity_poly_corrected, 'r-', label='Poly Corrected')
            plt.plot(wavelengths, intensity_snip_corrected, 'g-', label='SNIP Corrected')
            plt.plot(wavelengths, signal + noise, 'k--', label='True Signal + Noise') # Original signal without baseline
            plt.title('Baseline Corrected Data')
            plt.legend()
            plt.grid(True)

            # Modify subplot 3 to show multiple smoothing methods
            plt.subplot(3, 1, 3)
            plt.plot(wavelengths, intensity_with_nans, label='Raw Intensity (with NaNs)', alpha=0.3)
            plt.plot(wavelengths, smoothed_intensity, label=f'Smoothed (SavGol W=15, P=3)', color='purple', linewidth=1.5)
            plt.plot(wavelengths, smoothed_wavelet, label=f'Smoothed (Wavelet sym4)', color='orange', linewidth=1.5)
            plt.title('Smoothing Comparison (with NaN handling)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        except ImportError: print("\nMatplotlib not found. Skipping plots.")
        except Exception as e: print(f"\nError during plotting: {e}")
    else:
        print("PyWavelets not installed, skipping wavelet smoothing test.")