# -*- coding: utf-8 -*-
"""
Core functions for Calibration-Free LIBS (CF-LIBS) calculations.
Includes Boltzmann plot for temperature, Saha-Boltzmann for electron density,
and CF-LIBS concentration estimation.

Note: Saha-Boltzmann and CF-LIBS implementations herein are *simplified approximations*.
      For accurate quantitative results, consider implementing standard literature methods
      (e.g., full Saha-Boltzmann plots, standard CF-LIBS summation, optical thickness checks).
      Requires valid atomic data files loaded via core.atomic_data.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Set

# --- SciPy Imports ---
SCIPY_AVAILABLE = False
try:
    from scipy.stats import linregress
    SCIPY_AVAILABLE = True
except ImportError:
    logging.error("SciPy not found. CF-LIBS calculations requiring linregress will fail.")
    # Define dummy to avoid NameError, but it will raise ImportError if called
    def linregress(*args, **kwargs):
        raise ImportError("SciPy required for linear regression but is not installed.")

# Import data models and atomic data functions
from .data_models import Peak, NISTMatch
from .atomic_data import get_partition_function, get_ionization_energy

# --- Physical Constants ---
K_B_EV = 8.617333262e-5 # Boltzmann constant in eV/K
# Constants for Saha (SI units) - Ensure these are correctly used or switch to eV/cm^-3 formulation
H_EV_S = 4.135667696e-15 # Planck constant eV*s
M_E_KG = 9.1093837015e-31 # Electron mass kg
K_B_J = K_B_EV * 1.602176634e-19 # Boltzmann constant J/K
H_J_S = H_EV_S * 1.602176634e-19 # Planck constant J*s
# Saha pre-factor, units: m^-3 * K^(-3/2)
SAHA_FACTOR_SI = 2 * ( (2 * np.pi * M_E_KG * K_B_J) / (H_J_S**2) )**(3/2)
# Verify the exact formula and unit consistency before using SAHA_FACTOR_SI in Ne calculation! (See Issue 19.9)

# --- Boltzmann Plot Calculation ---
def calculate_boltzmann_temp(
    lines_data: pd.DataFrame
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[pd.DataFrame]]:
    """
    Calculates plasma temperature using the Boltzmann plot method.

    Args:
        lines_data (pd.DataFrame): DataFrame containing candidate lines with columns:
                                   'intensity', 'wavelength_nm', 'ei_upper', 'gi_upper', 'aki',
                                   and optionally 'label'.

    Returns:
        Tuple[Optional[float], Optional[float], Optional[float], Optional[pd.DataFrame]]:
            - Temperature (K) or None if calculation fails.
            - Temperature standard error (K) or None.
            - R-squared value of the fit or None.
            - DataFrame with columns ['x_energy_ev', 'y_boltzmann_term', 'label'] for plotting, or None.
    """
    if not SCIPY_AVAILABLE:
        logging.error("SciPy unavailable for Boltzmann plot calculation.")
        return None, None, None, None

    required_cols = ['intensity', 'wavelength_nm', 'ei_upper', 'gi_upper', 'aki']
    missing_cols = [c for c in required_cols if c not in lines_data.columns]
    if missing_cols:
        logging.error(f"Boltzmann plot failed: Input DataFrame missing required columns: {missing_cols}")
        return None, None, None, None

    # --- Data Preparation & Cleaning ---
    try:
        # Work on a copy
        df = lines_data[required_cols + ['label'] if 'label' in lines_data.columns else required_cols].copy()
        initial_rows = len(df)

        # Convert required columns to numeric, coercing errors to NaN
        for col in required_cols:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

        # Check for positive values *before* dropping NaNs for these critical columns
        # Intensity should reflect fitted area/amplitude, which must be positive.
        # Aki and gi (upper state g) must be positive. Wavelength is implicitly positive.
        positive_check_cols = ['intensity', 'wavelength_nm', 'gi_upper', 'aki']
        valid_mask = pd.Series(True, index=df.index) # Start with all true
        for col in positive_check_cols:
            valid_mask &= (df[col] > 1e-12) # Use small threshold > 0

        # Also check upper energy level is finite (can be zero or negative in principle, but usually positive)
        valid_mask &= df['ei_upper'].notna() & np.isfinite(df['ei_upper'])

        df = df[valid_mask]
        rows_after_pos_check = len(df)
        if rows_after_pos_check < initial_rows:
             logging.warning(f"Boltzmann: Removed {initial_rows - rows_after_pos_check} rows with non-positive intensity/Aki/gk or non-finite E_k.")

        # Drop any remaining rows with NaNs in required columns (should be fewer now)
        df.dropna(subset=required_cols, inplace=True)
        rows_after_dropna = len(df)

        if rows_after_dropna < 2:
            logging.error(f"Boltzmann plot requires at least 2 valid data points after cleaning, found {rows_after_dropna}.")
            return None, None, None, None

        # Calculate x and the argument for y safely
        df['x_energy_ev'] = df['ei_upper'] # x = Upper Energy Level E_k

        # Calculate argument for y = ln( I * lambda / (A_ki * g_k) )
        # Numerator and Denominator already checked > 0 above
        numerator = df['intensity'] * df['wavelength_nm']
        denominator = df['aki'] * df['gi_upper']

        # Avoid division by zero (already handled by positive check) and log(<=0)
        # Denominator was checked > 1e-12, Numerator > 1e-12 * Wavelength > 0
        # Calculate argument and check if it's positive before taking log
        log_argument = numerator / denominator
        log_valid_mask = log_argument > 1e-12 # Check argument is positive

        rows_before_log_check = len(df)
        df = df[log_valid_mask]
        rows_after_log_check = len(df)

        if rows_after_log_check < rows_before_log_check:
             logging.warning(f"Boltzmann: Removed {rows_before_log_check - rows_after_log_check} rows with non-positive argument for logarithm.")

        if len(df) < 2:
            logging.error(f"Boltzmann plot requires at least 2 valid data points after log argument check, found {len(df)}.")
            return None, None, None, None

        # Calculate y term using the filtered positive log argument
        df['y_boltzmann_term'] = np.log(log_argument[log_valid_mask])

        # Check for final NaN/Inf in coordinates (shouldn't happen if logic is right)
        x = df['x_energy_ev'].to_numpy()
        y = df['y_boltzmann_term'].to_numpy()
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            logging.error("Boltzmann plot coordinates (x or y) contained NaN/Inf values unexpectedly. Aborting fit.")
            return None, None, None, df[['x_energy_ev', 'y_boltzmann_term', 'label'] if 'label' in df else ['x_energy_ev', 'y_boltzmann_term']]

        num_points = len(x) # Number of points used in final fit
        logging.info(f"Performing Boltzmann fit using {num_points} valid data points.")

    except Exception as e:
        logging.error(f"Error during Boltzmann data preparation: {e}", exc_info=True)
        return None, None, None, None

    # --- Perform Linear Regression ---
    try:
        slope, intercept, r_val, p_val, stderr = linregress(x, y)
        r2 = r_val**2
        logging.info(f"Boltzmann fit: Slope={slope:.4f}, Intercept={intercept:.4f}, R²={r2:.4f}, StdErr={stderr:.4e}, P-value={p_val:.3e}")

        # Slope = -1 / (k_B * T_e)
        temp_k: Optional[float] = None
        temp_err: Optional[float] = None

        # Check slope sign and magnitude
        if slope >= -1e-9: # Allow very small negative slopes, but warn
             logging.warning(f"Boltzmann plot slope ({slope:.4f}) is non-negative or very close to zero. "
                             f"Temperature calculation will be unreliable or infinite.")
             # Return success=False by returning None for temperature
             temp_k = np.inf if slope > -1e-9 else None # Indicate infinite temp for near-zero negative slope? Or just None? Let's use None.
             temp_err = np.nan
             # Keep R² and plot data if available
             plot_df = df[['x_energy_ev', 'y_boltzmann_term', 'label'] if 'label' in df else ['x_energy_ev', 'y_boltzmann_term']]
             return temp_k, temp_err, r2, plot_df

        # Calculate Temperature and Error
        temp_k = -1.0 / (slope * K_B_EV)
        # Propagate error: dT = | (1 / (slope^2 * kB)) * dSlope | = | T / slope * dSlope |
        temp_err = abs(temp_k / slope * stderr) if np.isfinite(stderr) else np.nan

        # Check for very small slope magnitude leading to huge uncertainty
        # Define a threshold based on relative error or absolute slope value?
        # Let's warn if relative error > 100% or slope is extremely small
        rel_err_percent = (temp_err / temp_k * 100) if temp_k is not None and temp_k != 0 and temp_err is not None and np.isfinite(temp_err) else np.inf
        if abs(slope) < 1e-3 or rel_err_percent > 100: # Arbitrary threshold for small slope
             logging.warning(f"Boltzmann slope ({slope:.4e}) is very small or fit uncertainty is high ({rel_err_percent:.1f}%). "
                             f"Calculated temperature T={temp_k:.1f} ± {temp_err:.1f} K has high uncertainty.")

        logging.info(f"Boltzmann Result: Tₑ = {temp_k:.1f} ± {temp_err:.1f} K (R² = {r2:.4f})")
        plot_df = df[['x_energy_ev', 'y_boltzmann_term', 'label'] if 'label' in df else ['x_energy_ev', 'y_boltzmann_term']]
        return temp_k, temp_err, r2, plot_df

    except Exception as e:
        logging.error(f"Error during Boltzmann linear regression: {e}", exc_info=True)
        plot_df = df[['x_energy_ev', 'y_boltzmann_term', 'label'] if 'label' in df else ['x_energy_ev', 'y_boltzmann_term']]
        return None, None, None, plot_df

# --- Saha-Boltzmann for Electron Density ---
def calculate_electron_density_saha(
    lines_ion1: pd.DataFrame, # DataFrame for lower ionization stage lines
    lines_ion2: pd.DataFrame, # DataFrame for higher ionization stage lines
    temperature_k: float,
    species1_key: str,        # Normalized key, e.g., "Fe I"
    species2_key: str         # Normalized key, e.g., "Fe II"
) -> Optional[float]:
    """
    Estimates electron density (Ne) using the Saha-Boltzmann equation ratio method (Simplified Approximation).

    Args:
        lines_ion1 (pd.DataFrame): DataFrame of selected lines for the lower ionization stage.
        lines_ion2 (pd.DataFrame): DataFrame of selected lines for the higher ionization stage.
                                   Both require columns: 'intensity', 'aki', 'gi_upper', 'ei_upper'.
        temperature_k (float): Plasma temperature in Kelvin (from Boltzmann plot or other source).
        species1_key (str): Identifier for the lower ionization stage (e.g., "Fe I").
        species2_key (str): Identifier for the higher ionization stage (e.g., "Fe II").

    Returns:
        Optional[float]: Estimated electron density in cm^-3, or None if calculation fails.

    WARNING: This function uses a simplified method based on averaging log terms from line intensities.
             It does NOT perform a full Saha-Boltzmann plot analysis, does not verify LTE consistency
             between the ionization stages, and the formula used requires careful validation against
             literature for accuracy and unit consistency (Issue 19.9). Use results with caution.
    """
    logging.warning("Executing calculate_electron_density_saha using a SIMPLIFIED approximation. Results may be inaccurate.")
    logging.info(f"Attempting Saha Nₑ calculation: {species1_key}/{species2_key} @ T={temperature_k:.0f}K.")

    if lines_ion1.empty or lines_ion2.empty:
        logging.error("Saha: Input DataFrames for one or both species are empty.")
        return None
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        logging.error(f"Saha: Invalid Temperature Tₑ = {temperature_k}.")
        return None

    # --- Retrieve and Validate Required Atomic Data ---
    U1 = get_partition_function(species1_key, temperature_k)
    U2 = get_partition_function(species2_key, temperature_k)
    V_ion = get_ionization_energy(species1_key) # Ionization energy of the *lower* state

    if U1 is None or not np.isfinite(U1) or U1 <= 0:
        logging.error(f"Saha: Invalid or missing partition function U(T) for {species1_key}.")
        return None
    if U2 is None or not np.isfinite(U2) or U2 <= 0:
        logging.error(f"Saha: Invalid or missing partition function U(T) for {species2_key}.")
        return None
    if V_ion is None or not np.isfinite(V_ion) or V_ion <= 0:
        logging.error(f"Saha: Invalid or missing ionization energy V_ion for {species1_key}.")
        return None

    # --- Data Cleaning Helper (Inline) ---
    required_cols = ['intensity', 'aki', 'gi_upper', 'ei_upper'] # Using upper state E_k, g_k
    def clean_and_validate(df: pd.DataFrame, species_name: str) -> Optional[pd.DataFrame]:
        """Cleans DataFrame, checks columns, filters bad values."""
        if not all(c in df.columns for c in required_cols):
            logging.error(f"Saha: DataFrame for {species_name} missing required columns: "
                          f"{[c for c in required_cols if c not in df.columns]}")
            return None
        df_c = df[required_cols].copy()
        initial_rows = len(df_c)
        # Convert to numeric, coerce errors
        for col in required_cols:
            df_c.loc[:, col] = pd.to_numeric(df_c[col], errors='coerce')
        # Drop rows with NaN in essential numeric columns
        df_c.dropna(subset=required_cols, inplace=True)
        # Filter based on positivity
        for col in ['intensity', 'aki', 'gi_upper']: # E_k can be zero
            df_c = df_c[df_c[col] > 1e-12]
        rows_after_clean = len(df_c)
        if rows_after_clean < initial_rows:
             logging.debug(f"Saha cleaning for {species_name}: Removed {initial_rows - rows_after_clean} invalid/non-positive rows.")
        if df_c.empty:
             logging.error(f"Saha: No valid lines remaining for {species_name} after cleaning.")
             return None
        return df_c

    df1_clean = clean_and_validate(lines_ion1, species1_key)
    df2_clean = clean_and_validate(lines_ion2, species2_key)

    if df1_clean is None or df2_clean is None:
        return None # Error already logged

    # --- Simplified Calculation (Average Log Term Method) ---
    # See Issue 6.3 / 19.9 - THIS IS APPROXIMATE AND NEEDS VALIDATION
    kbt_ev = K_B_EV * temperature_k
    if kbt_ev <= 0: logging.error("Saha: k_B * T resulted in non-positive value."); return None

    try:
        # Calculate term = ln(I / (A_ki * g_k)) + E_k / (k_B * T) for each line
        # This term is proportional to ln(N_s / U_s) + constant
        # Ensure Intensity / (Aki * gk) is positive before log
        log_arg1 = df1_clean['intensity'] / (df1_clean['aki'] * df1_clean['gi_upper'])
        valid_log1 = log_arg1 > 1e-12
        if not valid_log1.all():
             logging.warning(f"Saha: Removing {np.sum(~valid_log1)} lines for {species1_key} with non-positive log argument.")
             df1_clean = df1_clean[valid_log1]
             log_arg1 = log_arg1[valid_log1]
        if df1_clean.empty: logging.error(f"Saha: No valid lines for {species1_key} after log check."); return None
        df1_clean['log_term'] = np.log(log_arg1) + df1_clean['ei_upper'] / kbt_ev

        log_arg2 = df2_clean['intensity'] / (df2_clean['aki'] * df2_clean['gi_upper'])
        valid_log2 = log_arg2 > 1e-12
        if not valid_log2.all():
             logging.warning(f"Saha: Removing {np.sum(~valid_log2)} lines for {species2_key} with non-positive log argument.")
             df2_clean = df2_clean[valid_log2]
             log_arg2 = log_arg2[valid_log2]
        if df2_clean.empty: logging.error(f"Saha: No valid lines for {species2_key} after log check."); return None
        df2_clean['log_term'] = np.log(log_arg2) + df2_clean['ei_upper'] / kbt_ev

        # Average these terms (using median for robustness)
        avg_log1 = df1_clean['log_term'].median()
        avg_log2 = df2_clean['log_term'].median()

        if not (np.isfinite(avg_log1) and np.isfinite(avg_log2)):
            logging.error(f"Saha: Calculation resulted in non-finite average log terms (Avg1={avg_log1}, Avg2={avg_log2}).")
            return None

        # Solve for ln(Ne) using the formula derived from Saha equation (VALIDATE THIS FORMULA AND UNITS!)
        # ln(Ne [m^-3]) = (avg_log1 - avg_log2) + ln(2*U2/U1) + ln(SAHA_FACTOR_SI) + 1.5*ln(T [K]) - (V_ion [eV] / kbt_ev)
        # WARNING: Unit consistency of ln(SAHA_FACTOR_SI) + 1.5*ln(T) term is suspect (Issue 19.9)
        # This part likely needs revision using a consistently derived formula.
        logging.warning("Saha Nₑ formula unit consistency needs careful validation (Issue 19.9).")
        term_ratio = avg_log1 - avg_log2
        term_U_ratio = np.log(2 * U2 / U1) # Factor 2 for electron g=2
        term_const_T = np.log(SAHA_FACTOR_SI) + 1.5 * np.log(temperature_k)
        term_Vion = V_ion / kbt_ev

        ln_Ne_m3 = term_ratio + term_U_ratio + term_const_T - term_Vion

        if not np.isfinite(ln_Ne_m3):
             logging.error(f"Saha: Calculation resulted in non-finite ln(Nₑ): Ratio={term_ratio:.2f}, U={term_U_ratio:.2f}, ConstT={term_const_T:.2f}, Vion={term_Vion:.2f}")
             return None

        Ne_m3 = np.exp(ln_Ne_m3) # Electron density in m^-3
        Ne_cm3 = Ne_m3 * 1e-6 # Convert to cm^-3

        if not np.isfinite(Ne_cm3) or Ne_cm3 <= 0:
            logging.error(f"Saha: Invalid final Nₑ calculated ({Ne_cm3:.3e} cm⁻³). Check input lines/data/formula.")
            return None

        logging.info(f"Estimated Nₑ (Saha Approx.): {Ne_cm3:.3e} cm⁻³ for {species1_key}/{species2_key}")
        return float(Ne_cm3)

    except Exception as e:
        logging.error(f"Error during Saha Nₑ calculation: {e}", exc_info=True)
        return None

# --- CF-LIBS Concentration ---
def _filter_lines_for_cflibs(
    peaks: List[Peak],
    max_delta_lambda_nm: float = 0.05,
    min_fit_r2: float = 0.90
) -> pd.DataFrame:
    """
    Filters peaks to select usable lines for CF-LIBS calculation based on fits and matches.

    Selects the best NIST match per peak based on wavelength proximity.
    Uses peak area if available and valid, otherwise falls back to amplitude.

    Args:
        peaks: List of Peak objects (must have fits and potential_matches populated).
        max_delta_lambda_nm: Maximum allowed difference between fitted/detected peak wavelength
                             and the NIST database wavelength for a match to be considered.
        min_fit_r2: Minimum R-squared value for the peak fit to be considered reliable.

    Returns:
        pd.DataFrame: DataFrame containing usable lines with necessary columns for CF-LIBS,
                      or an empty DataFrame if no usable lines are found.
                      Columns: 'element', 'species', 'ion_stage', 'wavelength_nm',
                               'intensity', 'aki', 'ei_upper', 'gi_upper'.
    """
    usable_lines = []
    # Required attributes from the NISTMatch object (ei=Ek, gi=gk - upper levels)
    required_atomic_keys = {'ei', 'gi', 'aki'}

    logging.debug(f"Filtering {len(peaks)} peaks for CF-LIBS (max Δλ={max_delta_lambda_nm}nm, min R²={min_fit_r2})...")
    peaks_no_fit = 0
    peaks_bad_fit = 0
    peaks_no_matches = 0
    matches_bad_atomic_data = 0
    matches_bad_intensity = 0
    matches_lambda_mismatch = 0

    for peak in peaks:
        # 1. Check for successful fit
        if not peak.best_fit or not peak.best_fit.success:
            peaks_no_fit += 1
            continue
        if peak.best_fit.r_squared is not None and peak.best_fit.r_squared < min_fit_r2:
            peaks_bad_fit += 1
            continue

        # 2. Check for potential NIST matches
        if not peak.potential_matches:
            peaks_no_matches += 1
            continue

        # 3. Select the best match based on wavelength difference
        best_match_for_peak: Optional[NISTMatch] = None
        min_diff = float('inf')
        for match in peak.potential_matches:
             if not hasattr(match, 'wavelength_db') or not np.isfinite(match.wavelength_db): continue
             diff = abs(peak.wavelength_fitted_or_detected - match.wavelength_db)
             if diff < min_diff:
                  min_diff = diff
                  best_match_for_peak = match

        # 4. Check if best match meets criteria
        if best_match_for_peak is None:
             # Should not happen if potential_matches is not empty, but safety check
             continue
        if min_diff > max_delta_lambda_nm:
             matches_lambda_mismatch += 1
             continue

        # 5. Check required atomic data in the best match
        atomic_data_valid = True
        atomic_data = {}
        for key in required_atomic_keys:
            val = getattr(best_match_for_peak, key, None)
            if val is None or not isinstance(val, (int, float)) or not np.isfinite(val) or val <= 0:
                 # Aki and gi must be positive
                 atomic_data_valid = False
                 break
            atomic_data[key] = val
        if not atomic_data_valid:
             matches_bad_atomic_data += 1
             continue

        # 6. Get Intensity (prefer Area, fallback to Amplitude)
        intensity_val = None
        if hasattr(peak.best_fit, 'area') and peak.best_fit.area is not None and np.isfinite(peak.best_fit.area) and peak.best_fit.area > 1e-12:
            intensity_val = peak.best_fit.area
            intensity_type = 'Area' # Track which was used (optional)
        elif hasattr(peak.best_fit, 'amplitude') and peak.best_fit.amplitude is not None and np.isfinite(peak.best_fit.amplitude) and peak.best_fit.amplitude > 1e-12:
            intensity_val = peak.best_fit.amplitude
            intensity_type = 'Amplitude'
            if not hasattr(peak.best_fit, 'area'):
                logging.warning("Peak fitting does not provide 'area'. Using 'amplitude' for CF-LIBS. "
                                "Area is generally preferred for quantitative analysis. (Needs implementation in peak_fitter)")
        else:
             matches_bad_intensity += 1
             continue # Skip if neither area nor amplitude is valid positive

        # 7. Check Optical Thinness (Placeholder)
        # TODO: Implement optical thinness checks here. This is CRUCIAL for CF-LIBS accuracy.
        #       Checks might involve filtering strong resonance lines, comparing line ratios,
        #       or using criteria based on line strength (Aki*gi) and lower level population.
        logging.warning(f"CF-LIBS Filter: Optical thinness check is NOT implemented for line {best_match_for_peak.element} {best_match_for_peak.ion_state_str} @ {best_match_for_peak.wavelength_db:.3f} nm. Results assume optical thinness.")


        # 8. Add to usable list if all checks pass
        usable_lines.append({
            'element': best_match_for_peak.element,
            'species': f"{best_match_for_peak.element} {best_match_for_peak.ion_state_str}",
            'ion_stage': best_match_for_peak.ion_state_int,
            'wavelength_nm': peak.wavelength_fitted_or_detected, # Use fitted wavelength
            'intensity': intensity_val, # Use valid Area or Amplitude
            'aki': atomic_data['aki'],
            'ei_upper': atomic_data['ei'], # Upper energy E_k
            'gi_upper': atomic_data['gi'], # Upper stat weight g_k
            # Add other useful info? Peak Index? Match DB Wavelength?
            'peak_index': peak.index,
            'db_wavelength_nm': best_match_for_peak.wavelength_db,
            'intensity_type': intensity_type,
        })

    # Log filtering summary
    logging.info(f"CF-LIBS Filter Summary: Input Peaks={len(peaks)}, Usable Lines={len(usable_lines)}")
    if peaks_no_fit > 0: logging.debug(f" - Skipped {peaks_no_fit} peaks (no successful fit).")
    if peaks_bad_fit > 0: logging.debug(f" - Skipped {peaks_bad_fit} peaks (fit R² < {min_fit_r2}).")
    if peaks_no_matches > 0: logging.debug(f" - Skipped {peaks_no_matches} peaks (no NIST matches).")
    if matches_lambda_mismatch > 0: logging.debug(f" - Skipped {matches_lambda_mismatch} matches (Δλ > {max_delta_lambda_nm} nm).")
    if matches_bad_atomic_data > 0: logging.debug(f" - Skipped {matches_bad_atomic_data} matches (missing/invalid E_k, g_k, or A_ki).")
    if matches_bad_intensity > 0: logging.debug(f" - Skipped {matches_bad_intensity} matches (invalid peak intensity/area).")

    if not usable_lines:
        logging.warning("CF-LIBS Filter: No usable lines found after filtering.")
        return pd.DataFrame() # Return empty DataFrame

    return pd.DataFrame(usable_lines)


def calculate_cf_libs_conc(
    peaks: List[Peak],
    temperature_k: float,
    electron_density_cm3: Optional[float] = None, # Currently unused
    max_delta_lambda_nm: float = 0.05,
    min_fit_r2: float = 0.90
) -> Optional[pd.DataFrame]:
    """
    Estimates elemental concentrations using the CF-LIBS method (Simplified Approximation).

    Assumes Local Thermodynamic Equilibrium (LTE) and Optically Thin plasma.
    Uses a simplified formula based on averaging line properties.

    Args:
        peaks: List of Peak objects with fits and NIST match correlations.
        temperature_k: Plasma temperature in Kelvin.
        electron_density_cm3: Electron density in cm^-3 (currently unused, for future models).
        max_delta_lambda_nm: Max wavelength difference for filtering lines.
        min_fit_r2: Minimum R-squared for filtering fitted lines.

    Returns:
        Optional[pd.DataFrame]: DataFrame with columns 'Element' and 'Concentration' (normalized),
                                or None if calculation fails.

    WARNING: This function uses a SIMPLIFIED approximation method. It does NOT perform
             standard CF-LIBS summation or rigorously handle potential issues like
             self-absorption or deviations from LTE. Use results with caution.
             Requires external atomic data files for partition functions.
    """
    logging.warning("Executing calculate_cf_libs_conc using a SIMPLIFIED approximation. Results may be inaccurate.")
    logging.info(f"Attempting CF-LIBS Concentration calculation @ T={temperature_k:.0f}K.")

    if not peaks:
        logging.error("CF-LIBS: Cannot calculate concentrations, no peaks provided.")
        return None
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        logging.error(f"CF-LIBS: Invalid Temperature Tₑ = {temperature_k}.")
        return None

    # 1. Filter peaks to get usable lines for calculation
    usable_df = _filter_lines_for_cflibs(peaks, max_delta_lambda_nm, min_fit_r2)
    if usable_df.empty:
        logging.error("CF-LIBS: No usable lines found after filtering. Cannot calculate concentrations.")
        return None
    logging.info(f"CF-LIBS: Using {len(usable_df)} filtered lines for calculation.")

    # 2. Check and Retrieve Partition Functions
    all_species = set(usable_df['species'])
    partition_funcs: Dict[str, Optional[float]] = {}
    missing_U = False
    for sp in all_species:
        U = get_partition_function(sp, temperature_k)
        if U is None or not np.isfinite(U) or U <= 0:
            logging.error(f"CF-LIBS: Invalid or missing partition function U(T={temperature_k:.0f}K) for species '{sp}'. Cannot proceed.")
            missing_U = True
            # Store None to indicate failure for this species
            partition_funcs[sp] = None
        else:
            partition_funcs[sp] = U

    if missing_U:
        # Do not proceed if essential U(T) data is missing
        return None

    # --- 3. Simplified Calculation (Average F_s + ln(U_s) per species) ---
    # See Issue 6.5 - THIS IS APPROXIMATE
    kbt_ev = K_B_EV * temperature_k
    if kbt_ev <= 0: logging.error("CF-LIBS: k_B * T resulted in non-positive value."); return None

    try:
        # Calculate Boltzmann term argument: exp(-E_k / kBT)
        exp_arg = -usable_df['ei_upper'] / kbt_ev
        # Clip argument to prevent overflow/underflow in exp()
        exp_arg = np.clip(exp_arg, -700, 700)
        boltzmann_term = np.exp(exp_arg)
        boltzmann_term = np.nan_to_num(boltzmann_term, nan=0.0, posinf=0.0, neginf=0.0) # Handle potential NaNs

        # Calculate argument for log term: ln(Intensity * Wavelength / (Aki * gk))
        log_arg_numerator = usable_df['intensity'] * usable_df['wavelength_nm']
        log_arg_denominator = usable_df['aki'] * usable_df['gi_upper']
        # Ensure argument is positive before taking log
        valid_log_mask = (log_arg_denominator > 1e-12) & (log_arg_numerator > 0)
        log_arg = np.full_like(log_arg_numerator, np.nan) # Initialize with NaN
        log_arg[valid_log_mask] = log_arg_numerator[valid_log_mask] / log_arg_denominator[valid_log_mask]
        log_arg[log_arg <= 1e-12] = np.nan # Set non-positive results to NaN

        log_term = np.log(log_arg) # Will produce NaN where log_arg was NaN

        # Combine terms: F_value = ln(I*lambda / (A*g)) + E_k / kBT (This is proportional to ln(C_s * Factor / U_s))
        # Note: E_k/kBT = -exp_arg
        usable_df['F_value'] = log_term - exp_arg # F_value = log_term + E_k/kBT

        # Drop rows where F_value calculation failed (due to NaN inputs or log issues)
        initial_f_rows = len(usable_df)
        usable_df.dropna(subset=['F_value'], inplace=True)
        if len(usable_df) < initial_f_rows:
             logging.warning(f"CF-LIBS: Removed {initial_f_rows - len(usable_df)} lines with invalid F_value during calculation.")

        if usable_df.empty:
            logging.error("CF-LIBS: No valid F_values calculated after cleaning. Cannot proceed.")
            return None

    except Exception as e:
        logging.error(f"CF-LIBS: Error calculating intermediate F_values: {e}", exc_info=True)
        return None

    # --- Calculate Average Q_s = F_avg + ln(U_s) per species ---
    species_Q: Dict[str, float] = {}
    species_lines_count: Dict[str, int] = {}
    # Group by species *after* calculating F_value
    grouped = usable_df.groupby('species')

    for species_name, group_df in grouped:
        F_vals = group_df['F_value'] # Already cleaned of NaNs
        if len(F_vals) == 0: continue # Skip if somehow group is empty

        # Use median F value for robustness? Or mean? Let's stick to mean for now.
        F_avg = F_vals.mean()
        U_s = partition_funcs.get(species_name) # Already validated U_s earlier

        if U_s is None: continue # Should not happen if validation worked

        Q_s = F_avg + np.log(U_s) # Q_s = Average( ln(I*lambda*exp(-Ek/kBT) / (A*g)) ) + ln(U_s)
                                 # Q_s = Average( ln(Factor * C_s / U_s) ) + ln(U_s) ??? -> Needs careful check

        if not np.isfinite(Q_s):
            logging.warning(f"CF-LIBS: Calculated non-finite Q_s for {species_name}. Skipping this species.")
            continue

        species_Q[species_name] = Q_s
        species_lines_count[species_name] = len(F_vals)
        logging.debug(f"CF-LIBS - Species: {species_name}, Lines Used: {len(F_vals)}, F_avg: {F_avg:.3f}, U(T): {U_s:.3f}, Q_s: {Q_s:.3f}")

    if not species_Q:
        logging.error("CF-LIBS: Failed to calculate Q_s for any species. Cannot determine concentrations.")
        return None

    # --- Estimate Concentrations C_s ≈ exp(Q_s) / Sum_all[exp(Q_s)] ---
    # This simplified normalization assumes the 'Factor' is constant across species, which is often not strictly true.
    exp_Q_vals = {sp: np.exp(Q) for sp, Q in species_Q.items() if np.isfinite(Q)}
    if not exp_Q_vals:
        logging.error("CF-LIBS: No finite exp(Q_s) values calculated.")
        return None

    total_exp_Q = sum(exp_Q_vals.values())
    if total_exp_Q <= 0 or not np.isfinite(total_exp_Q):
        logging.error(f"CF-LIBS: Invalid total sum of exp(Q_s) ({total_exp_Q}). Cannot normalize.")
        return None

    # --- Aggregate results by element ---
    element_concentration_fraction: Dict[str, float] = {}
    for species_name, exp_Q_value in exp_Q_vals.items():
        try:
            element = species_name.split()[0] # Extract element symbol
            species_fraction = exp_Q_value / total_exp_Q # Fraction relative to total signal factor
            element_concentration_fraction[element] = element_concentration_fraction.get(element, 0.0) + species_fraction
        except IndexError:
             logging.warning(f"Could not parse element from species name '{species_name}'. Skipping.")
             continue

    if not element_concentration_fraction:
        logging.error("CF-LIBS: Could not determine concentration fractions for any element.")
        return None

    # --- Normalize final elemental concentrations to 1 (or 100%) ---
    final_sum = sum(element_concentration_fraction.values())
    if final_sum <= 0 or not np.isfinite(final_sum):
         logging.error(f"CF-LIBS: Invalid sum of elemental fractions ({final_sum}). Cannot normalize.")
         return None

    results = [
        {'Element': el, 'Concentration': frac / final_sum} # Normalize to sum to 1
        for el, frac in element_concentration_fraction.items()
    ]
    results_df = pd.DataFrame(results).sort_values('Concentration', ascending=False).reset_index(drop=True)

    logging.info(f"CF-LIBS Concentration Estimation (Approximate) Result:\n{results_df.round(4)}")
    return results_df