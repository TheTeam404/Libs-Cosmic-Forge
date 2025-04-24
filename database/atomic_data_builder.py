# -*- coding: utf-8 -*-
"""
Builds atomic data files used by the LIBS Forge application.

Reads atomic energy level data from source files, calculates partition functions
U(T) over a range of temperatures, extracts ionization energies V_ion, and
saves the results into CSV files ('partition_functions.csv', 'ionization_energies.csv').

*** ============================= ***
***  CRITICAL USER ACTION NEEDED  ***
*** ============================= ***
This script contains **PLACEHOLDER** functions for parsing source files:
  - `parse_levels_file(filepath)`
  - `extract_ionization_energy(levels_df, filepath)`

You **MUST MODIFY** these two functions to correctly read and interpret the
structure and content of **YOUR specific input data files** (e.g., downloaded
NIST ASD level listings, files from other databases, custom formats).

The default placeholder functions **WILL NOT WORK** with real data and will
cause the script to fail or produce empty/incorrect output files. Refer to
the comments within these functions for guidance and examples.

Failure to correctly implement these parsers according to your data format
will lead to missing or inaccurate atomic data within the main LIBS Forge
application, rendering calculations like Saha-Boltzmann Nₑ or CF-LIBS
concentrations unreliable or impossible.
************************************

Potential Data Sources:
1. NIST Atomic Spectra Database (ASD): Download level data (HTML or ideally TSV/CSV if available). You will need to write parsing logic for the specific download format.
2. External Compilations: Using pre-compiled tables from scientific literature or other atomic databases. Adapt parsers to their format.

Workflow:
1. Obtain atomic level data files (containing at least energy levels and statistical weights 'g') for the elements/ions of interest. Ionization limits are also needed, either within the file or extracted separately.
2. Place these source files in a dedicated directory (default: 'database/source_atomic_levels/'). Use clear filenames like 'Fe_I.txt', 'Ca_II.dat' to help with species inference.
3. **CRITICAL:** Modify the `parse_levels_file` and `extract_ionization_energy` functions below in this script to match your source file format.
4. Run this script from the project root directory (the one containing `main.py`):
   `python database/atomic_data_builder.py [options]`
   Use `python database/atomic_data_builder.py --help` for options.
5. The script will attempt to process the files and generate/overwrite `partition_functions.csv` and `ionization_energies.csv` in the application's atomic data directory (default: `database/atomic_data/`).
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import traceback
from typing import List, Dict, Optional, Tuple, Set, Any

# --- Setup Project Root Path & Imports ---
# Ensures the script can find core modules when run directly
try:
    # Assumes this script is in 'database/' relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"INFO: Added project root '{project_root}' to sys.path.")

    # Import necessary paths and utilities AFTER potentially adding root to path
    from core.atomic_data import ATOMIC_DATA_DIR, PARTITION_FUNC_FILE, IONIZATION_ENERGY_FILE
    from utils.helpers import setup_logging, get_project_root
except ImportError as e:
    print(f"[CRITICAL ERROR] Failed to import necessary modules: {e}")
    print("Please ensure the script is run from the project root directory containing 'main.py'")
    print("or that the project structure is correct and accessible in the Python path.")
    sys.exit(1)
except Exception as e_setup:
    print(f"[CRITICAL ERROR] Unexpected error during initial setup: {e_setup}")
    traceback.print_exc()
    sys.exit(1)

# --- Constants ---
K_B_EV = 8.617333262e-5 # Boltzmann constant in eV/K
DEFAULT_TEMPERATURES_K = [ # Standard range for LIBS plasma temperatures
    3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 18000, 20000, 25000, 30000
]
# REQUIRED column names expected AFTER parsing in `parse_levels_file`
ENERGY_COL = 'Energy_eV' # Column containing energy level values in eV
G_WEIGHT_COL = 'g'       # Column containing statistical weight 'g'

# --- Argument Parser ---
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build atomic data files (U(T), V_ion) from source level data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-dir",
        default=os.path.join(get_project_root(), "database", "source_atomic_levels"),
        help="Directory containing source atomic level data files (e.g., one file per species)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=ATOMIC_DATA_DIR, # Use path defined in core.atomic_data
        help="Directory to save the generated atomic data CSV files ('partition_functions.csv', 'ionization_energies.csv')."
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite existing output CSV files if they exist."
    )
    parser.add_argument(
        "--temperatures",
        nargs='+',
        type=float,
        default=DEFAULT_TEMPERATURES_K,
        help="List of temperatures (K) for which to calculate partition functions."
    )
    parser.add_argument(
        "--log-level",
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level for console output."
    )
    return parser.parse_args()

# --- Core Calculation Functions ---

def calculate_partition_function(levels_df: pd.DataFrame, temperature_k: float) -> Optional[float]:
    """
    Calculates the internal partition function U(T) for a given temperature.

    Assumes the input DataFrame contains columns defined by ENERGY_COL ('Energy_eV')
    and G_WEIGHT_COL ('g').

    Note: This calculation includes all valid levels provided. For very high temperatures,
          the theoretical partition function sum might diverge depending on the atom/ion.
          This implementation does not explicitly handle potential divergence or apply
          cutoffs beyond filtering invalid input data. Ensure your input level data
          is appropriate for the target temperature range.

    Args:
        levels_df: DataFrame with energy levels and statistical weights.
        temperature_k: Temperature in Kelvin.

    Returns:
        The calculated partition function U(T), or None if calculation fails.
    """
    required_cols = {ENERGY_COL, G_WEIGHT_COL}
    if not required_cols.issubset(levels_df.columns):
        logging.error(f"Missing required columns for U(T) calculation. Need: {required_cols}, Got: {list(levels_df.columns)}")
        return None
    if not isinstance(temperature_k, (int, float)) or temperature_k <= 0:
        logging.error(f"Invalid temperature for U(T) calculation: {temperature_k}. Must be positive number.")
        return None

    try:
        # Ensure data is numeric, coercing errors to NaN
        energies = pd.to_numeric(levels_df[ENERGY_COL], errors='coerce')
        g_values = pd.to_numeric(levels_df[G_WEIGHT_COL], errors='coerce')

        # Filter out invalid entries (NaN energy/g, g <= 0)
        valid_mask = energies.notna() & g_values.notna() & (g_values > 0)
        if not valid_mask.any():
            logging.warning(f"No valid levels found for U(T) calculation at T={temperature_k:.0f}K after filtering.")
            return None # No valid levels to sum

        energies_valid = energies[valid_mask].to_numpy(dtype=float)
        g_valid = g_values[valid_mask].to_numpy(dtype=float)

        # Calculate Boltzmann factor argument, checking for numerical issues
        kbt_ev = K_B_EV * temperature_k
        if kbt_ev <= 0: # Should not happen if T > 0, but safety check
             logging.error(f"k_B * T calculation resulted in non-positive value ({kbt_ev:.3e}) at T={temperature_k:.0f}K.")
             return None

        exp_arg = -energies_valid / kbt_ev

        # Check for non-finite arguments before exponentiation
        if not np.all(np.isfinite(exp_arg)):
             num_non_finite = np.sum(~np.isfinite(exp_arg))
             logging.warning(f"Found {num_non_finite} non-finite exponent arguments at T={temperature_k:.0f}K. Treating their contribution as zero.")
             # Replace non-finite with large negative number -> exp -> 0
             exp_arg[~np.isfinite(exp_arg)] = -np.inf

        # Avoid overflow/underflow in exp() by clipping argument magnitude
        # exp(-700) is effectively zero, exp(700) is large but finite (~1e304)
        exp_arg = np.clip(exp_arg, -700, 700)

        boltzmann_factor = np.exp(exp_arg)

        # Handle potential NaNs/Infs resulting from exp (should be prevented by clipping/finite check)
        if not np.all(np.isfinite(boltzmann_factor)):
             num_non_finite_bf = np.sum(~np.isfinite(boltzmann_factor))
             logging.warning(f"Found {num_non_finite_bf} non-finite Boltzmann factors at T={temperature_k:.0f}K. Treating as zero.")
             boltzmann_factor = np.nan_to_num(boltzmann_factor, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate sum: U(T) = Sum[ g_i * exp(-E_i / (k_B * T)) ]
        partition_sum = np.sum(g_valid * boltzmann_factor)

        # Final check: Partition sum must be positive and finite
        if not np.isfinite(partition_sum) or partition_sum <= 1e-12: # Use a small threshold > 0
             logging.warning(f"Invalid partition sum calculated ({partition_sum:.3e}) at T={temperature_k:.0f}K. Check input levels/g-values.")
             return None

        return float(partition_sum)

    except Exception as e:
        logging.error(f"Error calculating U(T) at T={temperature_k:.0f}K: {e}", exc_info=True)
        return None


# --- USER MODIFICATION REQUIRED ---
def extract_ionization_energy(levels_df: pd.DataFrame, filepath: str) -> Optional[float]:
    """
    *** PLACEHOLDER: User MUST implement this function ***

    Extracts the ionization energy (V_ion) for the species from the source data file.
    The logic here is **ENTIRELY DEPENDENT** on how ionization energy information
    is stored in **your specific source file format**.

    Common Strategies:
    1. Keyword Search: Read the original file line by line, looking for keywords like
       "Ionization Energy", "Ionization Limit", "IP", etc., followed by a numerical value.
    2. Header Information: Check if the energy is stored in a specific header line.
    3. Max Energy Level: Sometimes the ionization limit is listed as the highest energy level
       in the levels table (often requires checking a specific flag or configuration notation).
    4. Separate File: Ionization energies might be in a completely separate reference file.

    Args:
        levels_df (pd.DataFrame): DataFrame containing the parsed level data. This might
                                  be useful if the limit is embedded within the level table.
        filepath (str): Path to the original source file being processed. This allows
                        reading the file directly to look for keywords or headers.

    Returns:
        Ionization energy in electron Volts (eV), or None if not found or extraction fails.
        Must return a positive floating point number.
    """
    # --- START OF USER MODIFICATION AREA ---
    # Example: Remove this error and implement your parsing logic.
    logging.critical("Function `extract_ionization_energy` requires implementation!")
    logging.critical(f"You MUST edit this function in 'database/atomic_data_builder.py' to parse ionization energy from YOUR file format for '{os.path.basename(filepath)}'.")
    raise NotImplementedError("Ionization energy extraction logic not implemented by user.")

    # --- Example Strategy 1: Keyword Search (Adapt keywords and parsing) ---
    # keywords_to_find = ["ionization energy", "ionization limit", "ionization potential"]
    # energy_found = None
    # try:
    #     with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
    #         for line_num, line in enumerate(f):
    #             line_lower = line.lower().strip()
    #             if any(kw in line_lower for kw in keywords_to_find):
    #                 logging.debug(f"Found potential keyword on line {line_num+1} of {os.path.basename(filepath)}: '{line.strip()}'")
    #                 # --- Add Your Parsing Logic Here ---
    #                 # Example: Try splitting by common delimiters and finding first number
    #                 parts = line.replace('=', ' ').replace(':', ' ').split()
    #                 for part in parts:
    #                     try:
    #                         # Attempt to convert part to float, handling potential units like 'eV'
    #                         potential_energy = float(part.lower().replace('ev','').strip())
    #                         if potential_energy > 0: # Basic sanity check
    #                             energy_found = potential_energy
    #                             logging.info(f"Extracted potential V_ion = {energy_found} eV from line: '{line.strip()}'")
    #                             break # Found a potential value on this line
    #                     except ValueError:
    #                         continue # Part wasn't a valid number
    #                 # --- End Your Parsing Logic ---
    #             if energy_found is not None:
    #                 break # Stop searching once found
    # except IOError as e:
    #     logging.error(f"Error reading file {filepath} to extract ionization energy: {e}")
    #     return None # File read error

    # --- Example Strategy 2: Max Energy (Check if applicable to your format) ---
    # if energy_found is None and ENERGY_COL in levels_df.columns and not levels_df.empty:
    #     try:
    #         max_energy = pd.to_numeric(levels_df[ENERGY_COL], errors='coerce').max()
    #         if pd.notna(max_energy) and max_energy > 0:
    #             # IMPORTANT: Add logic here to confirm this max energy IS the ionization limit
    #             # e.g., check a flag column, or assume based on data source documentation
    #             is_confirmed_limit = False # Replace with your confirmation logic
    #             if is_confirmed_limit:
    #                 logging.info(f"Using maximum energy level {max_energy} eV as V_ion (confirmation logic needed).")
    #                 energy_found = max_energy
    #             else:
    #                  logging.debug(f"Max energy {max_energy} eV found, but cannot confirm it's V_ion for {os.path.basename(filepath)}.")
    #     except Exception as e_max:
    #         logging.warning(f"Could not determine max energy for V_ion check: {e_max}")

    # --- Final Return ---
    # if energy_found is not None:
    #     return energy_found
    # else:
    #     logging.error(f"Failed to extract ionization energy for {os.path.basename(filepath)}. Check file content and parsing logic.")
    #     return None
    # --- END OF USER MODIFICATION AREA ---


# --- USER MODIFICATION REQUIRED ---
def parse_levels_file(filepath: str) -> Optional[pd.DataFrame]:
    """
    *** PLACEHOLDER: User MUST implement this function ***

    Parses a single source atomic levels data file into a pandas DataFrame.
    The logic here is **ENTIRELY DEPENDENT** on the format of **your source file**.

    The resulting DataFrame **MUST** contain columns named exactly:
      - 'Energy_eV': Containing the energy level values in electron Volts (eV).
      - 'g': Containing the statistical weight (degeneracy) 'g' for each level.

    Args:
        filepath (str): Path to the source levels data file.

    Returns:
        pd.DataFrame: DataFrame with minimally 'Energy_eV' and 'g' columns,
                      or None if parsing fails or required columns are missing/invalid.
    """
    # --- START OF USER MODIFICATION AREA ---
    # Example: Remove this error and implement your parsing logic.
    logging.critical("Function `parse_levels_file` requires implementation!")
    logging.critical(f"You MUST edit this function in 'database/atomic_data_builder.py' to parse YOUR file format for '{os.path.basename(filepath)}'.")
    raise NotImplementedError(f"Parsing logic required in `parse_levels_file` for {filepath}")

    # --- Example Parsing Logic for a Hypothetical NIST ASD TSV File ---
    # This is an EXAMPLE ONLY - Inspect your actual downloaded file structure.
    # NIST ASD download formats can vary. Look for options to download data,
    # potentially choosing tab-separated values (TSV) or comma-separated values (CSV).
    #
    # Assume the downloaded file looks something like this (TSV, comments start with '|'):
    # |-----------------------------------|
    # | NIST Atomic Spectra Database Data |
    # | Element: Fe I                     |
    # |-----------------------------------|
    # | Level (eV) | Configuration | Term | J | g | ... more columns ... |
    # |------------|---------------|------|---|---|----------------------|
    # |    0.000   | 3d6.4s2       | a5D  | 4 | 9 |                      |
    # |    0.052   | 3d6.4s2       | a5D  | 3 | 7 |                      |
    # |    0.098   | 3d6.4s2       | a5D  | 2 | 5 |                      |
    # ... etc ...

    # try:
    #     logging.debug(f"Attempting to parse '{os.path.basename(filepath)}' assuming NIST-like TSV format...")
    #     # Adjust skiprows, comment character, separator, and column names based on YOUR file
    #     df = pd.read_csv(
    #         filepath,
    #         sep='\t',         # Tab separator
    #         comment='|',      # Lines starting with '|' are comments
    #         skipinitialspace=True,
    #         header=0          # Assumes the first non-comment line is the header
    #     )
    #     logging.debug(f"Read {len(df)} rows from {os.path.basename(filepath)}. Columns found: {list(df.columns)}")

    #     # --- Find and Rename Required Columns (Case-Insensitive) ---
    #     # Define potential names for the columns we need in the source file
    #     energy_col_options = ['Level (eV)', 'Level', 'Energy', 'E (eV)']
    #     g_col_options = ['g', 'Stat. Weight', 'Weight']

    #     # Use the helper function to find the actual column names
    #     source_energy_col = _find_csv_column(df.columns, energy_col_options)
    #     source_g_col = _find_csv_column(df.columns, g_col_options)

    #     missing_required = []
    #     if not source_energy_col: missing_required.append(f"Energy ({energy_col_options})")
    #     if not source_g_col: missing_required.append(f"Stat. Weight ({g_col_options})")

    #     if missing_required:
    #         logging.error(f"Missing required source columns in '{os.path.basename(filepath)}': {', '.join(missing_required)}. Available columns: {list(df.columns)}")
    #         return None

    #     logging.debug(f"Mapping source columns: '{source_energy_col}' -> '{ENERGY_COL}', '{source_g_col}' -> '{G_WEIGHT_COL}'")

    #     # --- Select and Rename ---
    #     df_processed = df[[source_energy_col, source_g_col]].rename(columns={
    #         source_energy_col: ENERGY_COL,
    #         source_g_col: G_WEIGHT_COL
    #     })

    #     # --- Final Validation ---
    #     # Check if required columns exist after renaming (should always be true here)
    #     if not {ENERGY_COL, G_WEIGHT_COL}.issubset(df_processed.columns):
    #         logging.critical(f"Internal error: Renaming failed for {filepath}. Check column map.")
    #         return None # Should not happen if checks above passed

    #     # Check for non-numeric values BEFORE returning
    #     num_non_numeric_e = pd.to_numeric(df_processed[ENERGY_COL], errors='coerce').isna().sum()
    #     num_non_numeric_g = pd.to_numeric(df_processed[G_WEIGHT_COL], errors='coerce').isna().sum()
    #     if num_non_numeric_e > 0:
    #          logging.warning(f"Column '{ENERGY_COL}' in parsed {filepath} contained "
    #                          f"{num_non_numeric_e} non-numeric entries that will be coerced to NaN.")
    #     if num_non_numeric_g > 0:
    #          logging.warning(f"Column '{G_WEIGHT_COL}' in parsed {filepath} contained "
    #                          f"{num_non_numeric_g} non-numeric entries that will be coerced to NaN.")

    #     logging.info(f"Successfully parsed {len(df_processed)} levels from '{os.path.basename(filepath)}'.")
    #     return df_processed # Return the DataFrame with 'Energy_eV' and 'g' columns

    # except FileNotFoundError:
    #     logging.error(f"Source file not found: {filepath}")
    #     return None
    # except pd.errors.EmptyDataError:
    #      logging.error(f"Source file is empty or contains no parsable data: {filepath}")
    #      return None
    # except Exception as e:
    #     logging.error(f"Failed to parse levels file '{os.path.basename(filepath)}': {e}", exc_info=True)
    #     return None
    # --- End of Example Parsing Logic ---

    # --- If using the dummy data example from previous version: ---
    # if "dummy" in os.path.basename(filepath).lower():
    #      logging.info(f"Using dummy data for {filepath}")
    #      dummy_levels = {'Energy_eV': [0.0, 2.1, 3.5, 4.8, 10.2], 'g': [1, 3, 5, 7, 2]}
    #      df = pd.DataFrame(dummy_levels)
    #      return df
    # --- End of dummy data ---

    # --- Fallback if no logic implemented ---
    # return None # Return None if placeholder wasn't replaced
    # --- END OF USER MODIFICATION AREA ---


# --- File Processing Logic ---

def process_species_file(
    filepath: str,
    temperatures_k: List[float]
) -> Tuple[Optional[float], Optional[Dict[float, float]], int, int]:
    """
    Processes a single source file: parses levels, calculates V_ion and U(T).

    Args:
        filepath: Path to the source file.
        temperatures_k: List of temperatures (K) for U(T) calculation.

    Returns:
        Tuple containing:
        - Ionization energy (eV) or None.
        - Dictionary of {Temperature: U(T)} or None if none calculated.
        - Number of levels parsed (or 0).
        - Number of valid levels used for U(T) calculation (count from one temp).
    """
    logging.info(f"Processing source file: {os.path.basename(filepath)}")
    levels_df = parse_levels_file(filepath)

    if levels_df is None:
        # Error already logged in parse_levels_file
        logging.error(f"Skipping file due to parsing error: {os.path.basename(filepath)}")
        return None, None, 0, 0

    num_parsed_levels = len(levels_df)
    logging.debug(f"Parsed {num_parsed_levels} potential levels from {os.path.basename(filepath)}.")

    # --- Calculate Partition Functions ---
    partition_functions: Dict[float, float] = {}
    valid_levels_count_example = 0 # Store count from one temperature calculation
    # Pick a representative temperature (e.g., middle one) to check valid level count
    if temperatures_k:
         temp_for_count_check = temperatures_k[len(temperatures_k) // 2]
         first_calc = True
    else:
         temp_for_count_check = None
         first_calc = False

    for temp in temperatures_k:
        u_t = calculate_partition_function(levels_df, temp)
        if u_t is not None: # Only store valid results
            partition_functions[temp] = u_t
            # Get valid level count only once
            if first_calc and temp == temp_for_count_check:
                 try:
                     energies = pd.to_numeric(levels_df[ENERGY_COL], errors='coerce')
                     g_values = pd.to_numeric(levels_df[G_WEIGHT_COL], errors='coerce')
                     valid_mask = energies.notna() & g_values.notna() & (g_values > 0)
                     valid_levels_count_example = int(valid_mask.sum())
                 except Exception: pass # Ignore errors getting count
                 first_calc = False

    # --- Extract Ionization Energy ---
    ionization_energy = extract_ionization_energy(levels_df, filepath)

    # --- Logging Summary for the File ---
    if ionization_energy is None:
        logging.error(f"Could **NOT** extract ionization energy from {os.path.basename(filepath)}. Check file and parsing logic.")
    else:
         logging.info(f"Extracted Ionization Energy = {ionization_energy:.5f} eV for {os.path.basename(filepath)}")

    if not partition_functions:
        logging.warning(f"Could not calculate *any* valid partition functions for {os.path.basename(filepath)}. Check levels and g-values in the file.")
    else:
        num_temps_calc = len(partition_functions)
        logging.info(f"Calculated {num_temps_calc}/{len(temperatures_k)} partition functions for {os.path.basename(filepath)} "
                     f"(using ~{valid_levels_count_example}/{num_parsed_levels} valid levels).")

    # Return None for the dictionary if it's empty to make checks easier downstream
    u_t_result = partition_functions if partition_functions else None
    return ionization_energy, u_t_result, num_parsed_levels, valid_levels_count_example

# --- Species Inference ---
def infer_species_from_filename(filename: str) -> Optional[str]:
    """
    Attempts to infer the species name (e.g., "Fe I", "O II") from the filename.
    Provides warnings if inference is uncertain. Assumes formats like 'Fe_I.txt', 'Ca-II.dat'.

    Args:
        filename (str): The input filename.

    Returns:
        Optional[str]: The inferred species string (e.g., "Fe I") or None if failed.
    """
    if not filename: return None
    base_name = os.path.splitext(filename)[0]

    # Common suffixes to remove (case-insensitive)
    suffixes_to_remove = ['_levels', '-levels', '_data', '-data', '_asd']
    base_name_lower = base_name.lower()
    for suffix in suffixes_to_remove:
        if base_name_lower.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break # Remove only one suffix typically

    # Try splitting by common delimiters
    delimiters = ['_', '-']
    potential_species: Optional[str] = None

    for delim in delimiters:
        if delim in base_name:
            parts = base_name.split(delim)
            if len(parts) == 2:
                element, ion_stage = parts[0].strip(), parts[1].strip()
                # Validate parts: element is alphabetic, ion stage exists
                if element.isalpha() and ion_stage:
                    # Basic check for Roman numerals or digits
                    is_roman = all(c in 'IVXLCDM' for c in ion_stage.upper())
                    is_digit = ion_stage.isdigit()
                    if is_roman or is_digit:
                        # Format consistently: Capitalize element, uppercase Roman numeral
                        ion_stage_formatted = ion_stage.upper()
                        potential_species = f"{element.capitalize()} {ion_stage_formatted}"
                        break # Found a likely match

    if potential_species:
        logging.debug(f"Inferred species '{potential_species}' from filename '{filename}'")
        return potential_species
    else:
        # If no delimiter match, maybe the whole name is the species? (e.g., "FeI") - less reliable
        # Add more sophisticated regex matching here if needed based on common patterns
        logging.warning(f"Could not reliably infer species name from filename '{filename}'. "
                        "Recommend using formats like 'Element_IonStage.txt' (e.g., Fe_I.txt, Ca_II.dat) "
                        "for reliable automatic detection.")
        return None


# --- Main Execution ---
def main():
    """Main script execution function."""
    args = parse_arguments()

    # Setup logging based on args
    try:
        log_file = os.path.join(get_project_root(), "logs", "atomic_data_builder.log")
        # Ensure the log config dictionary is structured as expected by setup_logging
        log_config_dict = {'logging': {'log_level_console': args.log_level}}
        setup_logging(log_config_dict) # Use basic console level from args
    except NameError: # Handle case where setup_logging might not have been imported
        logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
        logging.error("Logging setup failed due to import errors. Using basic config.")
    except Exception as e_log_setup:
         logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
         logging.error(f"Error during logging setup: {e_log_setup}", exc_info=True)


    logging.warning("--- Starting Atomic Data Builder ---")
    logging.warning("*** CRITICAL: Ensure `parse_levels_file` and `extract_ionization_energy` functions ARE IMPLEMENTED for your data format! ***")
    logging.info(f"Input Directory (Source Levels): {args.input_dir}")
    logging.info(f"Output Directory (Generated CSVs): {args.output_dir}")
    logging.info(f"Temperatures for U(T) (K): {args.temperatures}")
    logging.info(f"Overwrite Existing Output: {args.overwrite}")

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logging.critical(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Ensure output directory exists
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        logging.critical(f"Could not create output directory {args.output_dir}: {e}")
        sys.exit(1)

    # Check if output files exist and handle overwrite logic BEFORE processing
    ion_path = os.path.join(args.output_dir, os.path.basename(IONIZATION_ENERGY_FILE))
    part_path = os.path.join(args.output_dir, os.path.basename(PARTITION_FUNC_FILE))

    skip_ion_save = False
    skip_part_save = False
    if not args.overwrite:
        if os.path.exists(ion_path):
             logging.warning(f"Output file '{ion_path}' already exists. Use --overwrite to replace it.")
             skip_ion_save = True
        if os.path.exists(part_path):
            logging.warning(f"Output file '{part_path}' already exists. Use --overwrite to replace it.")
            skip_part_save = True

    # Exit early if both files exist and overwrite is false
    if skip_ion_save and skip_part_save:
         logging.info("Both output files exist and --overwrite not specified. Nothing to do. Exiting.")
         sys.exit(0)

    # --- Process Files ---
    all_ionization_data: List[Dict[str, Any]] = []
    all_partition_data: List[Dict[str, Any]] = []
    processed_species: Set[str] = set() # Keep track of processed species to avoid duplicates
    file_count = 0
    success_count = 0 # Files from which *any* data was successfully extracted

    try:
        source_files = sorted([f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))])
    except FileNotFoundError:
         logging.critical(f"Input directory listed but cannot be accessed: {args.input_dir}")
         sys.exit(1)
    except Exception as e_list:
         logging.critical(f"Error listing files in input directory {args.input_dir}: {e_list}")
         sys.exit(1)


    if not source_files:
         logging.warning(f"No files found in input directory: {args.input_dir}")
         sys.exit(0)

    logging.info(f"Found {len(source_files)} files in input directory. Processing...")

    for filename in source_files:
        filepath = os.path.join(args.input_dir, filename)
        file_count += 1

        species = infer_species_from_filename(filename)
        if not species:
            logging.warning(f"Could not determine species for '{filename}', skipping file.")
            continue
        if species in processed_species:
            logging.warning(f"Species '{species}' already processed (likely duplicate filename pattern), skipping '{filename}'.")
            continue

        try:
            # Process the file: get V_ion and U(T) dict
            V_ion, U_T_dict, num_parsed, num_valid = process_species_file(filepath, args.temperatures)

            data_extracted_for_file = False
            # Add ionization energy if valid and saving is not skipped
            if V_ion is not None and np.isfinite(V_ion) and V_ion > 0:
                if not skip_ion_save:
                     all_ionization_data.append({'Species': species, 'IonizationEnergy_eV': V_ion})
                data_extracted_for_file = True # Mark as extracted even if save skipped

            # Add partition functions if valid and saving is not skipped
            if U_T_dict: # Check if dict is not None and not empty
                if not skip_part_save:
                    for temp, u_val in U_T_dict.items():
                        all_partition_data.append({
                            'Species': species,
                            'Temperature_K': temp,
                            'PartitionFunction_U': u_val
                        })
                data_extracted_for_file = True # Mark as extracted even if save skipped

            if data_extracted_for_file:
                success_count += 1
                processed_species.add(species) # Mark species as processed
            else:
                 # Log if parsing succeeded but no Vion or UT calculated
                 if num_parsed > 0:
                      logging.warning(f"Processed file '{filename}' (parsed {num_parsed} levels), but failed to extract valid V_ion or calculate any U(T).")

        except NotImplementedError as nie:
             # Catch the specific error from placeholder functions
             logging.critical(f"Stopping build process: Required function not implemented for file '{filename}': {nie}")
             sys.exit(1) # Exit immediately if placeholders are hit
        except Exception as e:
            logging.error(f"Unhandled error processing file '{filepath}': {e}", exc_info=True)
            # Continue to next file

    logging.info(f"Finished processing source files. Successfully extracted potential data for {success_count}/{file_count} files ({len(processed_species)} unique species).")

    # --- Save Results ---
    # Save Ionization Energies
    if all_ionization_data and not skip_ion_save:
        try:
            ion_df = pd.DataFrame(all_ionization_data).sort_values('Species').reset_index(drop=True)
            ion_df.to_csv(ion_path, index=False, float_format='%.6f') # Use more precision for V_ion
            logging.info(f"Saved Ionization Energies ({len(ion_df)} entries) to: {ion_path}")
        except Exception as e:
            logging.error(f"Failed to save Ionization Energy data to {ion_path}: {e}", exc_info=True)
    elif not all_ionization_data and not skip_ion_save:
        logging.warning("No Ionization Energy data was generated to save.")
    elif skip_ion_save:
        logging.info("Skipped saving Ionization Energy file (already exists and --overwrite not used).")

    # Save Partition Functions
    if all_partition_data and not skip_part_save:
        try:
            part_df = pd.DataFrame(all_partition_data).sort_values(['Species', 'Temperature_K']).reset_index(drop=True)
            # Use scientific notation for U(T) which can span large ranges
            part_df.to_csv(part_path, index=False, float_format='%.5e')
            logging.info(f"Saved Partition Functions ({len(part_df)} entries) to: {part_path}")
        except Exception as e:
            logging.error(f"Failed to save Partition Function data to {part_path}: {e}", exc_info=True)
    elif not all_partition_data and not skip_part_save:
        logging.warning("No Partition Function data was generated to save.")
    elif skip_part_save:
        logging.info("Skipped saving Partition Function file (already exists and --overwrite not used).")

    logging.info("--- Atomic Data Builder Finished ---")
    if success_count == 0 and file_count > 0:
         logging.critical("!!! CRITICAL: No data was successfully extracted from ANY source file. !!!")
         logging.critical("!!! This likely means the placeholder parsing functions were NOT implemented correctly. !!!")
         logging.critical("!!! Please review the output logs and EDIT `parse_levels_file` and `extract_ionization_energy` in this script. !!!")

if __name__ == "__main__":
    main()