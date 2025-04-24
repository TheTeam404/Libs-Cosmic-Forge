# -*- coding: utf-8 -*-
"""
Module for handling complex atomic data retrieval and calculation,
such as Partition Functions and Ionization Energies.

Loads data from user-specified CSV files (paths typically provided via config),
otherwise falls back to limited hardcoded defaults for ionization energies.
Provides functions to retrieve data, handling interpolation for partition functions.

IMPORTANT: The accuracy of CF-LIBS calculations heavily depends on the quality
           and completeness of the data in the specified partition function and
           ionization energy CSV files. Users MUST generate these files from
           reliable sources (e.g., NIST ASD website levels data, literature)
           using the provided placeholder builder script (`database/atomic_data_builder.py`)
           as a template, ensuring the format matches requirements (see builder script).
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any

# --- Constants ---
# Path constants removed - file paths should be passed in during loading

# --- Data Caches ---
# Store data keyed by normalized species string (e.g., "Fe I")
# Cache status: None (not attempted), True (load successful), False (load failed/file missing)
_partition_function_cache: Dict[str, Any] = {"loaded": None}
_ionization_energy_cache: Dict[str, Any] = {"loaded": None}

# --- Default/Fallback Data ---
# Limited set of default V_ion (eV) used ONLY if ionization_energies.csv is missing/fails to load.
# Keys MUST be normalized (e.g., "Fe I", not "fe i").
DEFAULT_IONIZATION_ENERGIES = {
    'H I': 13.59844, 'He I': 24.58739, 'Li I': 5.39172, 'Be I': 9.3227, 'B I': 8.29803,
    'C I': 11.2603, 'N I': 14.53414, 'O I': 13.61806, 'F I': 17.42282, 'Ne I': 21.56454,
    'Na I': 5.13908, 'Mg I': 7.64624, 'Al I': 5.98577, 'Si I': 8.15169, 'P I': 10.48669,
    'S I': 10.36001, 'Cl I': 12.96764, 'Ar I': 15.75962, 'K I': 4.34066, 'Ca I': 6.11316,
    'Fe I': 7.9024, 'Fe II': 16.1877, 'Ca II': 11.87172, 'Mg II': 15.03528, 'Al II': 18.82856,
    'Si II': 16.34585, 'Ti I': 6.8281, 'Ti II': 13.5755, 'Mn I': 7.43403, 'Mn II': 15.6399,
    'Ni I': 7.6398, 'Ni II': 18.16884, 'Cr I': 6.7665, 'Cr II': 16.4857,
    'Cu I': 7.72638, 'Cu II': 20.2924, 'Zn I': 9.3942, 'Zn II': 17.9644,
    'Sr I': 5.6949, 'Sr II': 11.03013, 'Ba I': 5.2117, 'Ba II': 10.00383,
    # Add more common elements if desired for improved fallback coverage
}

# --- Helper Functions ---

def _normalize_species_key(species: str) -> Optional[str]:
    """Normalizes a species string to 'Elem Ion' format (e.g., "fe i" -> "Fe I")."""
    if not isinstance(species, str): return None
    parts = species.strip().split()
    if len(parts) != 2: return None # Expect "Element Ion"
    element, ion_state = parts[0], parts[1]
    # Basic validation: element is alphabetic, ion_state is Roman numeral (or maybe integer later?)
    if not element.isalpha() or not ion_state: return None
    # Normalize: Capitalize element, uppercase Roman numeral ion state
    return f"{element.capitalize()} {ion_state.upper()}"

def _find_csv_column(df_columns: List[str], target_options: List[str]) -> Optional[str]:
    """Finds the best matching column name case-insensitively."""
    df_cols_lower = {col.lower().strip(): col for col in df_columns}
    for option in target_options:
        option_lower = option.lower().strip()
        if option_lower in df_cols_lower:
            return df_cols_lower[option_lower] # Return original case name
    return None

# --- Partition Function Handling ---

def _load_partition_functions(filepath: Optional[str] = None):
    """
    Loads partition function data U(T) from the specified CSV file.

    The CSV file should contain columns for 'Species' (e.g., "Fe I"),
    'Temperature_K', and 'PartitionFunction_U'.

    Args:
        filepath (Optional[str]): Full path to the partition function CSV file.
                                  If None, loading is skipped (cache remains unloaded).
    """
    global _partition_function_cache
    if _partition_function_cache.get("loaded") is not None: # Already attempted load
        return
    if filepath is None:
        logging.warning("No partition function file path provided. U(T) data will be unavailable.")
        _partition_function_cache = {"loaded": False}
        return

    # Ensure directory exists if path provided (though file should already exist)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    except Exception: pass # Ignore errors here, focus on file existence

    if not os.path.exists(filepath):
        logging.error(f"Partition function file not found: {filepath}. U(T) data unavailable.")
        _partition_function_cache = {"loaded": False}
        return

    try:
        logging.info(f"Loading U(T) from {filepath}...")
        df = pd.read_csv(filepath, comment='#') # Allow comments starting with #

        # Find columns flexibly using case-insensitive matching
        species_col = _find_csv_column(df.columns, ['Species', 'Ion', 'Element Spec'])
        temp_col = _find_csv_column(df.columns, ['Temperature_K', 'T(K)', 'Temperature', 'Temp'])
        u_col = _find_csv_column(df.columns, ['PartitionFunction_U', 'U(T)', 'Partition Function', 'U'])

        missing_required = []
        if not species_col: missing_required.append("'Species'")
        if not temp_col: missing_required.append("'Temperature_K'")
        if not u_col: missing_required.append("'PartitionFunction_U'")
        if missing_required:
             raise ValueError(f"File missing required columns: {', '.join(missing_required)}. Found: {list(df.columns)}")

        # Convert to numeric, handle errors, drop invalid rows
        df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
        df[u_col] = pd.to_numeric(df[u_col], errors='coerce')
        df.dropna(subset=[temp_col, u_col], inplace=True)

        # Validate: U(T) must be positive
        initial_rows = len(df)
        df = df[df[u_col] > 1e-12] # Filter out non-positive partition functions
        if len(df) < initial_rows:
             logging.warning(f"Removed {initial_rows - len(df)} rows with non-positive U(T) values from {os.path.basename(filepath)}.")

        # Group data by species and store sorted arrays in cache
        temp_cache: Dict[str, Dict[float, float]] = {}
        for _, row in df.iterrows():
            species = str(row[species_col]).strip()
            norm_species = _normalize_species_key(species) # Normalize key
            if norm_species is None:
                logging.warning(f"Skipping row with invalid species format '{species}' in {os.path.basename(filepath)}.")
                continue

            temp = float(row[temp_col])
            u_val = float(row[u_col])
            if norm_species not in temp_cache:
                temp_cache[norm_species] = {}
            temp_cache[norm_species][temp] = u_val

        # Clear previous cache (except 'loaded' flag) before repopulating
        loaded_status = _partition_function_cache.get("loaded")
        _partition_function_cache.clear()
        if loaded_status is not None: _partition_function_cache["loaded"] = loaded_status

        species_loaded_count = 0
        for species, data in temp_cache.items():
            sorted_temps = sorted(data.keys())
            if len(sorted_temps) >= 2: # Need at least 2 points for interpolation
                _partition_function_cache[species] = {
                    'temps': np.array(sorted_temps, dtype=float),
                    'u_values': np.array([data[t] for t in sorted_temps], dtype=float)
                }
                species_loaded_count += 1
            elif len(sorted_temps) == 1:
                # Store single point, but interpolation won't work
                _partition_function_cache[species] = {
                     'temps': np.array(sorted_temps, dtype=float),
                     'u_values': np.array([data[sorted_temps[0]]], dtype=float)
                }
                species_loaded_count += 1 # Count species even if only 1 point
                logging.warning(f"Only 1 data point found for U(T) for species '{species}'. Interpolation will not be possible.")
            # else: species had no valid data points

        _partition_function_cache["loaded"] = True
        logging.info(f"Loaded U(T) data for {species_loaded_count} species from {os.path.basename(filepath)}.")

    except FileNotFoundError: # Should be caught above, but safety
        logging.error(f"Partition function file not found during load attempt: {filepath}")
        _partition_function_cache = {"loaded": False}
    except ValueError as ve: # Catch specific errors like missing columns
         logging.error(f"Error loading partition functions from {filepath}: {ve}")
         _partition_function_cache = {"loaded": False}
    except pd.errors.ParserError as pe:
         logging.error(f"Error parsing partition function file {filepath} (check format/delimiter): {pe}")
         _partition_function_cache = {"loaded": False}
    except Exception as e:
        logging.error(f"Unexpected error loading partition functions from {filepath}: {e}", exc_info=True)
        _partition_function_cache = {"loaded": False}


def get_partition_function(species: str, temperature_k: float, filepath: Optional[str] = None) -> Optional[float]:
    """
    Retrieves or interpolates the partition function U(T) for a species at a given temperature.

    Args:
        species (str): The species identifier (e.g., "Fe I"). Case-insensitive matching attempted.
        temperature_k (float): The desired temperature in Kelvin.
        filepath (Optional[str]): Path to the partition function CSV file. If None and data
                                  is not cached, loading will fail. Should be provided on first call.

    Returns:
        Optional[float]: The partition function U(T), or None if data is unavailable,
                         temperature is outside the range, or interpolation fails.
                         **Does not extrapolate.**
    """
    global _partition_function_cache
    if _partition_function_cache.get("loaded") is None:
        _load_partition_functions(filepath) # Attempt load if not done yet

    if not _partition_function_cache.get("loaded", False):
        logging.debug(f"Partition function cache not loaded or failed to load. Cannot get U(T) for {species}.")
        return None

    norm_species = _normalize_species_key(species)
    if norm_species is None:
        logging.warning(f"Invalid species format '{species}' requested for U(T).")
        return None

    species_data = _partition_function_cache.get(norm_species)
    if not species_data:
        logging.warning(f"U(T) data not found in cache for '{norm_species}'.")
        return None

    temps = species_data.get('temps')
    u_values = species_data.get('u_values')

    if temps is None or u_values is None or len(temps) < 2:
        # Need >= 2 points for interpolation
        if len(temps) == 1 and np.isclose(temperature_k, temps[0]):
            logging.debug(f"Returning single stored U(T) point for {norm_species} at {temperature_k:.0f}K.")
            return float(u_values[0])
        else:
            logging.warning(f"Cannot interpolate U(T) for '{norm_species}': Need >= 2 data points, found {len(temps)}.")
            return None

    # --- Check if temperature is within the loaded range ---
    min_temp, max_temp = temps[0], temps[-1]
    if not (min_temp <= temperature_k <= max_temp):
        logging.error(f"Requested temperature T={temperature_k:.0f}K is outside the loaded range "
                      f"[{min_temp:.0f}-{max_temp:.0f}]K for species '{norm_species}'. Extrapolation not supported.")
        return None # Return None if outside range

    # --- Perform Linear Interpolation ---
    try:
        interpolated_u = np.interp(temperature_k, temps, u_values)
        if not np.isfinite(interpolated_u) or interpolated_u <= 0:
             logging.error(f"Interpolated U(T) for {norm_species} @ {temperature_k:.0f}K is invalid ({interpolated_u:.3e}).")
             return None
        logging.debug(f"Interpolated U(T={temperature_k:.0f}K) for {norm_species}: {interpolated_u:.3f}")
        return float(interpolated_u)
    except Exception as e:
        logging.error(f"Error during U(T) interpolation for {norm_species} @ {temperature_k}K: {e}", exc_info=True)
        return None

# --- Ionization Energy Handling ---

def _load_ionization_energies(filepath: Optional[str] = None):
    """
    Loads ionization energy data (V_ion) from the specified CSV file.
    If the file is missing or fails to load, falls back to hardcoded defaults.

    The CSV file should contain columns for 'Species' (e.g., "Fe I" - the lower state)
    and 'IonizationEnergy_eV'.

    Args:
        filepath (Optional[str]): Full path to the ionization energy CSV file.
                                  If None, uses defaults.
    """
    global _ionization_energy_cache
    if _ionization_energy_cache.get("loaded") is not None: # Already attempted load
        return

    file_was_loaded = False
    loaded_energies: Dict[str, float] = {}
    loaded_count = 0

    if filepath is None:
        logging.warning("No ionization energy file path provided. Using limited built-in defaults.")
    else:
        # Ensure directory exists if path provided
        try:
             os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except Exception: pass

        if not os.path.exists(filepath):
            logging.warning(f"Ionization energy file not found: {filepath}. Using limited built-in defaults.")
        else:
            try:
                logging.info(f"Loading V_ion from {filepath}...")
                df = pd.read_csv(filepath, comment='#')

                # Find columns flexibly
                species_col = _find_csv_column(df.columns, ['Species', 'Ion', 'Element Spec Lower', 'Lower Species'])
                energy_col = _find_csv_column(df.columns, ['IonizationEnergy_eV', 'V_ion (eV)', 'Ionization Energy (eV)'])

                missing_required = []
                if not species_col: missing_required.append("'Species'")
                if not energy_col: missing_required.append("'IonizationEnergy_eV'")
                if missing_required:
                    raise ValueError(f"File missing required columns: {', '.join(missing_required)}. Found: {list(df.columns)}")

                df[energy_col] = pd.to_numeric(df[energy_col], errors='coerce')
                df.dropna(subset=[energy_col], inplace=True)

                for _, row in df.iterrows():
                    species = str(row[species_col]).strip()
                    norm_species = _normalize_species_key(species) # Normalize key
                    if norm_species is None:
                        logging.warning(f"Skipping row with invalid species format '{species}' in {os.path.basename(filepath)}.")
                        continue

                    energy = float(row[energy_col])
                    if np.isfinite(energy) and energy > 0: # Basic validation
                        loaded_energies[norm_species] = energy
                        loaded_count += 1
                    else:
                        logging.warning(f"Skipping invalid ionization energy value '{energy}' for species '{species}' in {os.path.basename(filepath)}.")

                file_was_loaded = True # Mark that we attempted to load from file
                logging.info(f"Loaded {loaded_count} valid V_ion entries from {os.path.basename(filepath)}.")

            except FileNotFoundError: # Should be caught above, but safety
                 logging.warning(f"Ionization energy file not found during load attempt: {filepath}. Using limited built-in defaults.")
            except ValueError as ve: # Catch specific errors like missing columns
                 logging.error(f"Error loading ionization energies from {filepath}: {ve}. Using limited built-in defaults.")
            except pd.errors.ParserError as pe:
                 logging.error(f"Error parsing ionization energy file {filepath} (check format/delimiter): {pe}. Using limited built-in defaults.")
            except Exception as e:
                logging.error(f"Unexpected error loading ionization energies from {filepath}: {e}. Using limited built-in defaults.", exc_info=True)

    # --- Initialize Cache ---
    # Start with defaults, then update/overwrite with values loaded from file
    _ionization_energy_cache = DEFAULT_IONIZATION_ENERGIES.copy()
    if file_was_loaded and loaded_energies:
         _ionization_energy_cache.update(loaded_energies)
         logging.debug(f"Updated cache with {len(loaded_energies)} V_ion values from file.")
    elif not file_was_loaded:
         logging.debug("Initialized V_ion cache using only built-in defaults.")
    else: # File was loaded but resulted in no valid data
         logging.warning("File loaded for V_ion but contained no valid entries. Using only built-in defaults.")

    # --- Validate Final Cache Values ---
    invalid_defaults_count = 0
    for key, value in _ionization_energy_cache.items():
        if not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
             logging.warning(f"Cached ionization energy for '{key}' is invalid ({value}). It will be ignored.")
             # Optionally remove invalid entry? Or just let get_ionization_energy handle it.
             invalid_defaults_count += 1
    if invalid_defaults_count > 0:
         logging.warning(f"Found {invalid_defaults_count} invalid entries during V_ion cache validation.")

    _ionization_energy_cache["loaded"] = True # Mark cache as loaded (even if only defaults)


def get_ionization_energy(species_lower: str, filepath: Optional[str] = None) -> Optional[float]:
    """
    Retrieves the ionization energy (V_ion in eV) for the specified lower ionization stage.

    Args:
        species_lower (str): The species identifier for the *lower* ionization state
                             (e.g., "Fe I"). Case-insensitive matching attempted.
        filepath (Optional[str]): Path to the ionization energy CSV file. If None and data
                                  is not cached, loading will be attempted using defaults.
                                  Should be provided on first call.

    Returns:
        Optional[float]: Ionization energy in eV, or None if not found or invalid.
    """
    if _ionization_energy_cache.get("loaded") is None:
        _load_ionization_energies(filepath) # Attempt load if not done yet

    norm_species = _normalize_species_key(species_lower)
    if norm_species is None:
        logging.warning(f"Invalid species format '{species_lower}' requested for V_ion.")
        return None

    energy = _ionization_energy_cache.get(norm_species)

    if energy is None:
        # Changed from warning to error as this indicates missing essential data
        logging.error(f"Ionization energy (V_ion) not found for '{norm_species}'. "
                      "Check input file or defaults.")
        return None
    elif not isinstance(energy, (float, int)) or not np.isfinite(energy) or energy <= 0:
        logging.warning(f"Invalid V_ion ({energy}) found in cache for '{norm_species}'.")
        return None

    logging.debug(f"Retrieved V_ion for {norm_species}: {energy:.5f} eV")
    return float(energy)