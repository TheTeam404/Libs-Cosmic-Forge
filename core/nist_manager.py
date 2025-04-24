# -*- coding: utf-8 -*-
"""
Handles interaction with NIST Atomic Spectra Database using Astroquery for online searching.
Also provides helper functions for fetching data for specific elements/ions.
"""

import logging
import time
import traceback # Keep for potential detailed logging if needed
import numpy as np
from typing import List, Dict, Any, Optional

# --- Astroquery Imports ---
# Encapsulated to handle potential ImportError gracefully.
ASTROQUERY_AVAILABLE = False
Nist = None
u = None
Table = None
try:
    from astropy import units as u
    from astroquery.nist import Nist
    from astropy.table import Table
    # Removed MaskedColumn import as direct check isn't strictly needed with safe access
    ASTROQUERY_AVAILABLE = True
    logging.info("Astroquery library found. Online NIST search enabled.")
except ImportError:
    logging.warning("Astroquery or Astropy not found. Online NIST search will be disabled. "
                    "Install with 'pip install astroquery'.")

# Import data models
try:
    # Use relative import assuming data_models is in the same 'core' directory
    from .data_models import NISTMatch
except ImportError as e_import:
    logging.critical(f"CRITICAL ERROR in nist_manager.py: Cannot import NISTMatch data model: {e_import}.")
    raise ImportError(f"Core NISTMatch model failed to import in nist_manager: {e_import}") from e_import

# --- Constants ---
# Defaults moved to function signatures

# --- Online NIST Search Function ---

def search_online_nist(wavelength_nm: float,
                       tolerance_nm: float = 0.1, # Default directly in signature
                       query_delay_s: float = 1.0,  # Default directly in signature
                       timeout_s: float = 20       # Default directly in signature
                       ) -> List[NISTMatch]:
    """
    Performs an online query to the NIST ASD for lines within a wavelength range.

    Args:
        wavelength_nm (float): The center wavelength (in nanometers) to search around.
        tolerance_nm (float): The search tolerance (+/- nm) around the center wavelength.
        query_delay_s (float): Delay *before* executing the query (seconds).
        timeout_s (float): Timeout duration for the query (seconds).

    Returns:
        List[NISTMatch]: A list of NISTMatch objects found, sorted by proximity to the
                         query wavelength. Returns empty list on error or if none found.

    Note:
        Relies on `astroquery` library. Performance depends on NIST server responsiveness.
        Consider managing the astroquery version in requirements.txt for stability,
        as NIST column names can occasionally change.
    """
    if not ASTROQUERY_AVAILABLE:
        logging.error("Cannot perform online NIST search: Astroquery library not available.")
        return []

    if not np.isfinite(wavelength_nm) or not np.isfinite(tolerance_nm) or tolerance_nm <= 0:
        logging.error(f"Invalid search parameters for online NIST search: wavelength={wavelength_nm}, tolerance={tolerance_nm}")
        return []

    # Apply delay before the query to be nice to the NIST server
    if query_delay_s > 0:
        logging.debug(f"Applying {query_delay_s:.1f}s delay before NIST query.")
        time.sleep(query_delay_s)

    # Define search range in Angstroms (Astroquery/NIST prefers AA)
    min_wave_aa = (wavelength_nm - tolerance_nm) * 10.0 * u.AA
    max_wave_aa = (wavelength_nm + tolerance_nm) * 10.0 * u.AA

    logging.info(f"Querying NIST online: {wavelength_nm:.3f} ± {tolerance_nm:.3f} nm ({min_wave_aa:.2f} - {max_wave_aa:.2f})")

    matches: List[NISTMatch] = []
    try:
        # Execute the query - Using vacuum wavelengths is generally preferred for consistency
        table = Nist.query(min_wave_aa, max_wave_aa,
                           energy_level_unit="eV", # Request energy levels in eV
                           output_order="wavelength", # Sort results by wavelength
                           wavelength_type="vacuum", # Prioritize vacuum wavelengths
                           timeout=timeout_s)

        if table is None:
            # This case might indicate a timeout or connection error handled internally by astroquery
            logging.warning(f"NIST query for {wavelength_nm:.3f} nm returned None (potential timeout or connection issue).")
            return []
        if len(table) == 0:
            logging.info(f"No lines found online in NIST ASD for range around {wavelength_nm:.3f} nm.")
            return []

        logging.info(f"NIST query returned {len(table)} potential lines near {wavelength_nm:.3f} nm.")

        # --- Define expected NIST column names ---
        # Prioritize Ritz Vacuum, then Observed Vacuum, etc. Ritz are generally more precise.
        wav_col_priority = [ "Observed Wavelength Ritz VAC (nm)", "Observed Wavelength VAC (nm)",
                             "Observed Wavelength Ritz AIR (nm)", "Observed Wavelength AIR (nm)",
                             "Observed" ] # Last fallback column name
        # Core atomic data columns needed for calculations/identification
        aki_col = "Aki (s^-1)"
        ek_col = "Ek (eV)" # Upper energy level (E_k) -> Our ei for Boltzmann
        gk_col = "gk"      # Upper statistical weight (g_k) -> Our gi for Boltzmann
        ei_col = "Ei (eV)" # Lower energy level (E_i) -> Our ek
        gi_col = "gi"      # Lower statistical weight (g_i) -> Our gk
        elem_col = "Element"
        spec_col = "Spectrum" # e.g., "Fe I", "Ca II"
        # Store essential columns for pre-loop check
        essential_cols = {aki_col, ek_col, gk_col, elem_col, spec_col}


        # Process each row in the returned table
        processed_count = 0
        skipped_count = 0
        for i, row in enumerate(table):
            try:
                # --- Check if essential columns exist in this row (defensive check) ---
                row_cols = set(row.colnames)
                if not essential_cols.issubset(row_cols):
                     missing = essential_cols - row_cols
                     logging.warning(f"Skipping row {i}: Missing essential NIST columns: {missing}")
                     skipped_count += 1
                     continue

                # --- Extract Wavelength ---
                obs_wav_nm = None
                for col in wav_col_priority:
                    # Check column exists AND is not masked AND can be converted
                    if col in row_cols and row[col] is not np.ma.masked:
                         try:
                             val = float(row[col])
                             if np.isfinite(val):
                                  obs_wav_nm = val
                                  break # Found valid wavelength
                         except (ValueError, TypeError):
                             pass # Failed conversion, try next priority column
                if obs_wav_nm is None:
                     logging.debug(f"Skipping row {i}: No valid wavelength found in priority columns.")
                     skipped_count += 1
                     continue # Skip row if no valid wavelength found

                # --- Secondary Check: Ensure wavelength is reasonably within tolerance ---
                # NIST sometimes returns lines just outside the exact queried range.
                # The 1.01 factor provides a small buffer.
                if abs(obs_wav_nm - wavelength_nm) > tolerance_nm * 1.01:
                    logging.debug(f"Skipping row {i}: Wavelength {obs_wav_nm:.4f} nm outside effective tolerance ({tolerance_nm * 1.01:.4f} nm) of target {wavelength_nm:.4f} nm.")
                    skipped_count += 1
                    continue

                # --- Extract Other Parameters Safely ---
                def get_val(col_name: str, dtype=float, default=None):
                    """Helper to safely extract and convert potentially masked data."""
                    if col_name in row_cols and row[col_name] is not np.ma.masked:
                        try: return dtype(row[col_name])
                        except (ValueError, TypeError): return default
                    return default

                element = get_val(elem_col, str)
                spectrum_str = get_val(spec_col, str)
                # Check essential identifiers again after extraction
                if not element or not spectrum_str:
                     logging.debug(f"Skipping row {i}: Missing Element ('{element}') or Spectrum ('{spectrum_str}').")
                     skipped_count += 1
                     continue

                aki = get_val(aki_col, float)
                ei_upper = get_val(ek_col, float) # Upper E_k
                gi_upper = get_val(gk_col, float) # Upper g_k
                ei_lower = get_val(ei_col, float) # Lower E_i
                gi_lower = get_val(gi_col, float) # Lower g_i

                # --- Validate Required Atomic Data for CF-LIBS/Boltzmann ---
                # Check essential data: Aki > 0, Ek finite, gk > 0
                required_atomic_vals = [aki, ei_upper, gi_upper]
                if None in required_atomic_vals or not np.all(np.isfinite(required_atomic_vals)):
                    logging.debug(f"Skipping match {spectrum_str}@{obs_wav_nm:.4f}: Missing/invalid essential atomic data (Aki, E_k, g_k).")
                    skipped_count += 1
                    continue
                # Physical constraints: Aki and g must be positive
                if aki <= 1e-12 or gi_upper <= 1e-12: # Use small positive threshold
                     logging.debug(f"Skipping match {spectrum_str}@{obs_wav_nm:.4f}: Non-positive Aki ({aki:.2e}) or g_k ({gi_upper}).")
                     skipped_count += 1
                     continue

                # --- Parse Spectrum String for Ion State ---
                # Simple split assumes "Elem Ion" format (e.g., "Fe I", "Ca II")
                parts = spectrum_str.split()
                ion_state_str = "?"
                if len(parts) == 2:
                    ion_state_str = parts[1] # Assumes second part is ion state
                elif len(parts) == 1 and parts[0] == element:
                    # Handle neutral atoms sometimes represented just by element symbol (e.g., "H")
                    ion_state_str = "I"
                    spectrum_str = f"{element} I" # Standardize format
                    logging.debug(f"Interpreted single part spectrum '{parts[0]}' as '{spectrum_str}'.")

                # --- Create NISTMatch object ---
                # Note: Stores NIST UPPER level E/g (Ek/gk) as ei/gi for Boltzmann convenience
                match = NISTMatch(
                    element=element,
                    ion_state_str=ion_state_str,
                    wavelength_db=obs_wav_nm,
                    aki=aki,
                    ei=ei_upper, # NIST Ek -> Our ei (Upper E)
                    gi=gi_upper, # NIST gk -> Our gi (Upper g)
                    ek=ei_lower, # NIST Ei -> Our ek (Lower E)
                    gk=gi_lower, # NIST gi -> Our gk (Lower g)
                    # Generate a default label if needed
                    line_label=f"{spectrum_str} {obs_wav_nm:.4f}", # Use 4 decimals for NIST λ
                    source='NIST Online'
                )
                matches.append(match)
                processed_count += 1

            except (AttributeError, ValueError, TypeError, KeyError) as row_err:
                 # Log error specific to row processing but continue with next row
                 logging.warning(f"Skipping online row {i} due to data extraction/conversion error: {row_err}. Data: {dict(row)}", exc_info=False)
                 skipped_count += 1
                 continue
            except Exception as row_err:
                # Catch any other unexpected error during row processing
                logging.error(f"Unexpected error processing online row {i}: {row_err}. Data: {dict(row)}", exc_info=True)
                skipped_count += 1
                continue

    except ImportError: # Should be caught by ASTROQUERY_AVAILABLE check, but safety net
        logging.error("Astroquery import failed during online query execution.")
        return []
    except Exception as e:
        # Catch errors during the Nist.query call itself (e.g., network timeout not handled by astroquery)
        logging.error(f"Error during online NIST query execution: {e}", exc_info=True)
        return []

    # Sort final list of valid matches by proximity to the query wavelength
    matches.sort(key=lambda m: abs(m.wavelength_db - wavelength_nm))
    log_msg = f"Processed {processed_count} valid matches"
    if skipped_count > 0:
        log_msg += f" (skipped {skipped_count} rows)"
    log_msg += f" from online query near {wavelength_nm:.3f} nm."
    logging.info(log_msg)
    return matches


# --- Helper Function for Fetching Script ---
def get_nist_element_ion_data(element: str, ion_stage: int, timeout_s: float = 30) -> Optional[Table]:
    """
    Fetches NIST ASD data for a specific element and ionization stage across a broad range.
    Helper function used by the nist_data_fetcher.py script.

    Args:
        element (str): Element symbol (e.g., "Fe").
        ion_stage (int): Ionization stage (1 for Neutral, 2 for Singly Ionized, etc.).
        timeout_s (float): Timeout for the NIST query.

    Returns:
        Optional[astropy.table.Table]: Table of NIST data, or None if unavailable/error.
    """
    if not ASTROQUERY_AVAILABLE:
        logging.error("Cannot fetch NIST data: Astroquery not available.")
        return None
    if not isinstance(element, str) or not element or not isinstance(ion_stage, int) or ion_stage < 1:
        logging.error(f"Invalid element ('{element}') or ion stage ({ion_stage}) for NIST query.")
        return None

    # Define broad wavelength range (Angstroms) to query most lines
    # NIST ASD typically covers ~1 Angstrom to ~100 um (1e6 Angstroms)
    min_wave_aa = 1.0 * u.AA
    max_wave_aa = 1000000.0 * u.AA # 100 um

    # Construct the spectrum query string (e.g., "Fe I", "Ca II")
    roman_map_rev = {1:'I', 2:'II', 3:'III', 4:'IV', 5:'V', 6:'VI', 7:'VII', 8:'VIII', 9:'IX', 10:'X'} # Extend if needed
    roman_stage = roman_map_rev.get(ion_stage)
    if roman_stage is None:
        logging.error(f"Cannot convert ion stage {ion_stage} to Roman numeral for NIST query.")
        return None
    spectrum_query = f"{element.capitalize()} {roman_stage}"

    logging.info(f"Querying NIST online for ALL data for: {spectrum_query} (Timeout: {timeout_s}s)")

    try:
        # Query using line_matching requires spectrum argument
        table = Nist.query(min_wave_aa, max_wave_aa,
                           spectrum=spectrum_query, # Specify the element and ion stage
                           energy_level_unit="eV",
                           output_order="wavelength",
                           wavelength_type="vacuum", # Prefer vacuum wavelengths
                           timeout=timeout_s)

        if table is None:
            logging.warning(f"NIST query for {spectrum_query} returned None (potential timeout or connection issue).")
            return None
        if len(table) == 0:
            logging.info(f"No lines found in NIST ASD for {spectrum_query} in the queried range.")
            # Return None for consistency, indicating no data found
            return None

        logging.info(f"NIST query for {spectrum_query} successful, returned {len(table)} lines.")
        return table

    except ImportError:
        logging.error("Astroquery import failed during fetch execution.")
        return None
    except Exception as e:
        logging.error(f"Error during NIST query for {spectrum_query}: {e}", exc_info=True)
        return None

# --- Local Search Placeholder (Not Implemented) ---
# def search_local_nist(...) -> List[NISTMatch]:
#     logging.warning("Local NIST DB search is not implemented.")
#     return []