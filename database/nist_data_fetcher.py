# -*- coding: utf-8 -*-
"""
Script to fetch NIST Atomic Spectra Database data for specified elements and
ionization stages using astroquery and save it locally as CSV files.

This script iterates through elements and ionization states, queries NIST online,
and saves the results in a structured directory (default: database/nist_data_cache/).

*** WARNING: RESPONSIBLE USAGE REQUIRED ***
Running this for many elements and high ionization stages can take a VERY
long time (hours or days) and generate a large amount of data. It also places
significant load on the NIST ASD servers.

Please use responsibly:
- Fetch only the elements you actually need.
- Start with a lower `--max-ion` setting (e.g., 3 or 4).
- Consider increasing the `--delay` between requests (e.g., 3-5 seconds).
- Start with fewer `--workers` (e.g., 1 or 2).
- Check NIST's usage policies if performing very large bulk downloads.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

# --- Setup Project Root Path ---
# Ensure the script can find core modules when run directly
try:
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent # Assumes script is in database/
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"INFO: Added project root '{project_root}' to sys.path.")
except Exception as e_path:
    print(f"[ERROR] Could not determine project root: {e_path}")
    project_root = Path(".") # Fallback

# --- Local Imports ---
# Check for core dependencies first
_CORE_AVAILABLE = False
try:
    # Import setup_logging if enhanced logging configuration is desired
    # from utils.helpers import get_project_root, setup_logging
    from utils.helpers import get_project_root # get_project_root is needed for default output path
    _CORE_AVAILABLE = True
except ImportError:
    # Fallback basic logging if utils not found
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-7s - %(message)s', datefmt='%H:%M:%S')
    logging.warning("Could not import from utils.helpers. Using basic console logging.")
    # Define get_project_root locally if needed
    if 'get_project_root' not in globals():
        get_project_root = lambda: project_root # Simple lambda fallback

# Now import nist_manager, checking its critical dependency (astroquery)
try:
    from core.nist_manager import get_nist_element_ion_data, ASTROQUERY_AVAILABLE
    if not ASTROQUERY_AVAILABLE:
        logging.critical("Astroquery library is required but not installed or failed to import. Please run:")
        logging.critical("  pip install astroquery")
        sys.exit(1)
except ImportError as e:
    logging.critical(f"Error importing core modules (nist_manager): {e}")
    logging.critical("Please ensure the script is run from within the project structure "
                     "or the project root is in the Python path.")
    sys.exit(1)
except Exception as e_core:
     logging.critical(f"Unexpected error importing core modules: {e_core}", exc_info=True)
     sys.exit(1)

# --- Constants ---
DEFAULT_MAX_EMPTY_CONSECUTIVE = 3
DEFAULT_QUERY_DELAY_S = 2.5 # Increased default delay
DEFAULT_MAX_WORKERS = 2     # Reduced default workers
DEFAULT_OUTPUT_DIR = get_project_root() / "database" / "nist_data_cache"

# --- List of Common Elements (Fallback) ---
COMMON_LIBS_ELEMENTS = ["H","Li","Be","B","C","N","O","F","Na","Mg","Al","Si","P","S","Cl","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","U"]

# --- Roman Numeral Map (Extended) ---
ROMAN_MAP_REV = {
    1:'I', 2:'II', 3:'III', 4:'IV', 5:'V', 6:'VI', 7:'VII', 8:'VIII', 9:'IX', 10:'X',
    11:'XI', 12:'XII', 13:'XIII', 14:'XIV', 15:'XV', 16:'XVI', 17:'XVII', 18:'XVIII', 19:'XIX', 20:'XX'
}

# --- Argument Parser ---
def parse_arguments() -> argparse.Namespace:
    # Add warning to description
    parser = argparse.ArgumentParser(
        description=(
            "Fetch NIST ASD data locally. WARNING: Use responsibly! Check NIST usage policies. "
            "Fetching all elements/ions can take hours/days and load NIST servers. "
            "Consider fetching only needed elements and starting with low --max-ion/--workers and higher --delay."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-e", "--elements", nargs='+', default=None,
        help="List of specific element symbols (e.g., Fe Ca Si). Default: Uses a built-in list of common LIBS elements."
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Base directory to save element subdirectories containing CSV files."
    )
    parser.add_argument(
        "--max-ion", type=int, default=7,
        help="Maximum ionization stage to fetch (e.g., 7 for VII)."
    )
    parser.add_argument(
        "--max-empty", type=int, default=DEFAULT_MAX_EMPTY_CONSECUTIVE,
        help="Stop fetching higher ion stages for an element after N consecutive empty/error results."
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_QUERY_DELAY_S,
        help="Delay (s) applied *before* each query within a worker thread."
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_MAX_WORKERS,
        help="Maximum number of parallel download threads."
    )
    parser.add_argument(
        "--overwrite", action='store_true',
        help="Overwrite existing CSV files if they already exist."
    )
    parser.add_argument(
        "--log-level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level for console output."
    )
    return parser.parse_args()

# --- Worker Function ---
def fetch_and_save_worker(
    element: str,
    ion_stage: int,
    element_dir: Path,
    delay: float,
    overwrite: bool,
    timeout: float = 30.0 # Default timeout for individual query
) -> Tuple[str, int, str, Optional[int]]:
    """
    Worker task: fetches data for one element/ion stage, saves it, and returns status.

    Args:
        element: Element symbol.
        ion_stage: Ionization stage (1=I, 2=II, ...).
        element_dir: Path object for the element's output directory.
        delay: Time delay (seconds) before executing the query.
        overwrite: Boolean indicating whether to overwrite existing files.
        timeout: Timeout in seconds for the astroquery call.

    Returns:
        Tuple[str, int, str, Optional[int]]:
            (element, ion_stage, status_code, lines_saved_count)
            status_code: 'OK', 'Exists', 'NoData', 'Error'
            lines_saved_count: Number of lines saved, or None if not applicable.
    """
    status = 'Error'
    lines_saved = None
    roman = ROMAN_MAP_REV.get(ion_stage)
    if not roman:
        logging.warning(f"Cannot convert ion stage {ion_stage} to Roman for element {element}. Skipping.")
        return element, ion_stage, status, lines_saved

    species_name = f"{element} {roman}"
    filename = f"{element}_{roman}.csv"
    output_path = element_dir / filename

    # Check if file exists and overwrite is False
    if output_path.exists() and not overwrite:
        logging.info(f"Skipping {species_name}: File exists at '{output_path}' and overwrite=False.")
        status = 'Exists'
        return element, ion_stage, status, lines_saved # Treat existing as success for counting

    # Apply delay before query
    if delay > 0:
        logging.debug(f"Worker for {species_name} applying delay: {delay:.1f}s")
        time.sleep(delay)

    # Fetch data
    try:
        logging.debug(f"Worker querying NIST for {species_name}...")
        table = get_nist_element_ion_data(element, ion_stage, timeout_s=timeout)

        if table is not None and len(table) > 0:
            lines_saved = len(table)
            # Ensure directory exists before writing
            element_dir.mkdir(parents=True, exist_ok=True)
            # Write the table to CSV
            table.write(output_path, format="csv", overwrite=True) # Overwrite within worker if flag was passed
            logging.info(f"Saved: {species_name} ({lines_saved} lines) -> '{filename}'")
            status = 'OK'
        else:
            # Handle case where query succeeded but returned no data
            logging.info(f"No data found in NIST ASD for {species_name}.")
            status = 'NoData'
            # Optionally create an empty file marker? Or just log it.

    except FileNotFoundError as e_fnf: # Error during table.write if directory fails creation?
        logging.error(f"File system error saving data for {species_name} to '{output_path}': {e_fnf}")
        status = 'Error'
    except IOError as e_io: # Other file writing errors
        logging.error(f"I/O error saving data for {species_name} to '{output_path}': {e_io}")
        status = 'Error'
    except Exception as e: # Catch errors from get_nist_element_ion_data or table.write
        logging.error(f"Error processing {species_name}: {e}", exc_info=True)
        status = 'Error'

    return element, ion_stage, status, lines_saved

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_arguments()

    # Setup Logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    # Use basic config, assumes no file logging needed for this script by default
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)-7s - %(message)s', datefmt='%H:%M:%S')
    # Optional: If file logging desired, uncomment below and ensure setup_logging available
    # if _CORE_AVAILABLE:
    #    log_file = Path("logs") / "nist_data_fetcher.log" # Example path
    #    log_file.parent.mkdir(exist_ok=True)
    #    setup_logging({'logging': {'log_level_console': args.log_level, 'log_level_file': 'DEBUG'}}, log_file=str(log_file))

    elements_to_fetch = args.elements or COMMON_LIBS_ELEMENTS
    output_dir = args.output_dir.resolve() # Ensure absolute path

    logging.warning("--- Starting NIST Data Fetcher ---")
    logging.warning("*** Please Use Responsibly - Check NIST Usage Policies ***")
    logging.info(f"Elements to fetch: {', '.join(elements_to_fetch)}")
    logging.info(f"Max Ion Stage: {args.max_ion}")
    logging.info(f"Output Base Directory: {output_dir}")
    logging.info(f"Overwrite Existing Files: {args.overwrite}")
    logging.info(f"Query Delay per Worker: {args.delay}s")
    logging.info(f"Max Parallel Workers: {args.workers}")
    print("-" * 30, flush=True)

    # Validate output directory writability
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Try creating a temporary test file
        test_file = output_dir / ".permission_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e_perm:
        logging.critical(f"Output directory '{output_dir}' is not writable or accessible: {e_perm}")
        sys.exit(1)

    # --- Task Generation and Execution ---
    total_tasks = 0
    element_empty_counts = {el: 0 for el in elements_to_fetch}
    tasks_to_submit: List[Tuple] = []

    for element in elements_to_fetch:
        element_output_dir = output_dir / element # Use Path object
        for ion_stage in range(1, args.max_ion + 1):
             tasks_to_submit.append((element, ion_stage, element_output_dir))
             total_tasks += 1

    logging.info(f"Generated {total_tasks} potential download tasks.")
    if total_tasks == 0:
         logging.info("No tasks to execute. Exiting.")
         sys.exit(0)

    # --- Thread Pool Execution ---
    tasks_processed = 0
    results_summary = {'OK': 0, 'Exists': 0, 'NoData': 0, 'Error': 0}
    # Keep track of elements where max_empty is reached to avoid over-logging
    max_empty_reached_elements = set()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_and_save_worker, element, ion, el_dir, args.delay, args.overwrite): (element, ion)
            for element, ion, el_dir in tasks_to_submit
        }

        logging.info(f"Submitting {len(futures)} tasks to {args.workers} workers...")

        for future in as_completed(futures):
            element, ion_stage = futures[future]
            tasks_processed += 1
            progress_percent = (tasks_processed / total_tasks) * 100 if total_tasks > 0 else 0
            print(f"Progress: {tasks_processed}/{total_tasks} ({progress_percent:.1f}%) - Last processed: {element} {ROMAN_MAP_REV.get(ion_stage, '?')}", end='\r', flush=True)

            try:
                elem_res, ion_res, status_code, lines_count = future.result()

                # Check if max empty has been reached for this element *before* processing result
                if element_empty_counts.get(elem_res, 0) >= args.max_empty:
                    if elem_res not in max_empty_reached_elements:
                         logging.info(f"Max empty/error count ({args.max_empty}) reached for element {elem_res}. Further 'NoData'/'Error' results for this element may not be counted accurately in summary due to parallel execution.")
                         max_empty_reached_elements.add(elem_res)
                    # Even if reached, still count 'OK' or 'Exists' status
                    if status_code in ['OK', 'Exists']:
                        results_summary[status_code] += 1
                        element_empty_counts[elem_res] = 0 # Reset counter on success/exists
                    # Ignore NoData/Error for counting if threshold already met
                    elif status_code in ['NoData', 'Error'] and elem_res not in max_empty_reached_elements:
                         results_summary[status_code] += 1
                         element_empty_counts[elem_res] += 1
                    continue # Skip further processing for this result if threshold met

                # If threshold not met, process normally
                results_summary[status_code] += 1
                if status_code in ['OK', 'Exists']:
                    element_empty_counts[elem_res] = 0 # Reset count on success
                elif status_code in ['NoData', 'Error']:
                    element_empty_counts[elem_res] += 1 # Increment empty/error count

            except Exception as e_future:
                 # Error retrieving result from future itself
                 logging.error(f"Critical error getting result for future task ({element} {ion_stage}): {e_future}", exc_info=True)
                 results_summary['Error'] += 1
                 element_empty_counts[element] += 1 # Count as error for stopping logic

    # --- Final Summary ---
    print("\n" + "="*30) # Newline after progress indicator
    logging.warning("--- NIST Data Fetcher Finished ---")
    logging.info(f"Total Tasks Processed: {tasks_processed}/{total_tasks}")
    logging.info(f"Files Saved/Updated: {results_summary['OK']}")
    logging.info(f"Files Already Existed (Skipped): {results_summary['Exists']}")
    logging.info(f"Queries with No Data Returned: {results_summary['NoData']}")
    logging.info(f"Errors During Fetch/Save: {results_summary['Error']}")
    print("="*30)