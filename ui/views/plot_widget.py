# --- START OF REFACTORED FILE libs_cosmic_forge/ui/views/plot_widget.py ---
"""
Custom Matplotlib plot widget integrated with PyQt6 for displaying spectra,
peaks, fits, NIST lines, and handling interactive elements like highlighting
and annotations on hover/click.
"""
import logging
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt # Required for colormap access (plt.get_cmap)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text as MplText # Use explicit type hint alias
from matplotlib.artist import Artist
from matplotlib.legend import Legend # For type hinting legend
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, pyqtBoundSignal
# Use string literals for data model type hints (e.g., 'Spectrum') within THIS file.
# This is the standard way to resolve type checker errors caused by complex import
# situations or circular dependencies without affecting runtime behavior.
from typing import List, Optional, Dict, Any, Tuple, Union

# Ensure using QtAgg backend for compatibility with PyQt6
matplotlib.use('QtAgg')

# --- Project Root Calculation (Attempt) ---
# Tries to find the project root to add it to sys.path for local imports.
# This is often necessary if the package isn't installed in the environment.
# Consider improving project structure or using an editable install (pip install -e .)
# to make this unnecessary.
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _VIEWS_DIR = os.path.dirname(_SCRIPT_DIR)
    _UI_DIR = os.path.dirname(_VIEWS_DIR)
    _LIBS_COSMIC_FORGE_DIR = os.path.dirname(_UI_DIR)
    _PROJECT_ROOT = os.path.dirname(_LIBS_COSMIC_FORGE_DIR)

    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
        logging.debug(f"PlotWidget: Added project root '{_PROJECT_ROOT}' to sys.path for imports.")
except NameError:
    _PROJECT_ROOT = os.getcwd() # Fallback if __file__ is not defined
    logging.warning(f"PlotWidget: Could not determine project root via __file__. Using cwd '{_PROJECT_ROOT}'. Relative imports might fail.")
except Exception as e:
     _PROJECT_ROOT = os.getcwd() # Fallback on any other error
     logging.error(f"PlotWidget: Error calculating project root: {e}. Using cwd '{_PROJECT_ROOT}'.", exc_info=True)


# --- Import Core Components with Fallbacks ---
# These imports are REQUIRED for the widget's functionality at RUNTIME.
# The try-except block allows the UI to partially load or be tested even if
# these core modules are missing or cause import errors.
try:
    # Import the actual classes needed for runtime checks and operations
    from core.data_models import Spectrum, Peak, FitResult, NISTMatch
    # Import the fitting functions
    from core.processing import gaussian, lorentzian, pseudo_voigt
    CORE_MODULES_AVAILABLE = True
    logging.debug("PlotWidget: Core data models and processing functions imported successfully.")
except ImportError as e:
    logging.error(f"PlotWidget: CRITICAL - Failed to import core modules: {e}. Plotting will be severely limited. Using dummy placeholders.")
    # Define dummy placeholders if imports fail. This allows the class definition
    # to succeed but functionality relying on these types will be broken.
    class Spectrum: pass # type: ignore
    class Peak: pass # type: ignore
    class FitResult: pass # type: ignore
    class NISTMatch: pass # type: ignore
    def gaussian(*args, **kwargs): return np.zeros_like(args[0]) if len(args) > 0 and isinstance(args[0], np.ndarray) else None # type: ignore
    def lorentzian(*args, **kwargs): return np.zeros_like(args[0]) if len(args) > 0 and isinstance(args[0], np.ndarray) else None # type: ignore
    def pseudo_voigt(*args, **kwargs): return np.zeros_like(args[0]) if len(args) > 0 and isinstance(args[0], np.ndarray) else None # type: ignore
    CORE_MODULES_AVAILABLE = False
except Exception as e:
    # Catch other potential exceptions during import
     logging.critical(f"PlotWidget: An unexpected error occurred during core module import: {e}", exc_info=True)
     Spectrum = Peak = FitResult = NISTMatch = object # type: ignore
     gaussian = lorentzian = pseudo_voigt = lambda *args, **kwargs: None # type: ignore
     CORE_MODULES_AVAILABLE = False

# --- Type Hint Aliases ---
PlotElement = Optional[Artist] # Type alias for optional Matplotlib artists
FitLineDict = Dict[Tuple[int, str], Line2D] # Type alias for the dictionary storing fit lines


class SpectrumPlotWidget(QWidget):
    """
    A Matplotlib plotting widget embedded in PyQt6 for visualizing spectral data.

    Features:
    - Plots raw, processed spectra and baselines.
    - Displays detected and fitted peak markers.
    - Shows individual fit profile lines.
    - Visualizes NIST database matches.
    - Interactive hovering for data point/peak information.
    - Click-to-select peaks.
    - Theming support based on configuration.
    """

    # Signal emitted when a peak marker (scatter point) is clicked.
    # The integer payload is the original index of the clicked peak in the
    # list provided to plot_peaks().
    peak_clicked: pyqtBoundSignal = pyqtSignal(int)

    # --- Initialization ---
    # ***** CORRECTED PARAMETER ORDER *****
    def __init__(self, parent: Optional[QWidget] = None, config: Optional[Dict] = None):
        """
        Initializes the SpectrumPlotWidget.

        Args:
            parent (Optional[QWidget]): The parent widget in the Qt hierarchy.
            config (Optional[Dict]): Application configuration dictionary, expected
                                      to contain 'style' and 'plotting' sections.
        """
        # ***** Call super().__init__ with the PARENT widget *****
        super().__init__(parent)

        # Store the config dictionary
        self.config = config if config is not None else {}
        logging.info("Initializing SpectrumPlotWidget.")
        if not CORE_MODULES_AVAILABLE:
             logging.warning("PlotWidget initialized, but core modules failed to import. Functionality will be limited.")

        # --- Matplotlib Figure and Canvas Setup ---
        self.figure = Figure(tight_layout=True) # Enable tight layout for better spacing
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self) # Standard MPL navigation toolbar

        # --- Plot Element Management ---
        # Dictionary to store references to plotted Matplotlib artists.
        # Keys are descriptive strings, values are the artist objects or None.
        self._plot_elements: Dict[str, Union[PlotElement, FitLineDict]] = {
            'raw': None,        # Line2D for raw spectrum
            'proc': None,       # Line2D for processed spectrum
            'base': None,       # Line2D for baseline
            'det': None,        # PathCollection for detected peak markers
            'fit': None,        # PathCollection for fitted peak markers
            'fits': {},         # Dict mapping (peak_idx, profile_name) -> Line2D for fit lines
            'nist_lines': None, # LineCollection for NIST match vertical lines
            'highlight': None,  # PathCollection for the highlighted peak marker
            'legend': None      # Matplotlib Legend object
        }
        # Separate list for NIST text annotations as they are added/removed alongside nist_lines
        self._nist_annotations: List[MplText] = []

        # --- Data References ---
        # Store references to the *data objects* currently plotted. Necessary for interactions
        # like finding peak info on hover/click. Using STRING LITERALS for type hints.
        self._peaks_ref: List['Peak'] = []                 # Reference to the list of plotted peaks
        self._matches_ref: List['NISTMatch'] = []          # Reference to the list of plotted NIST matches
        self._highlighted_peak_list_index: Optional[int] = None # Index (in _peaks_ref) of the currently highlighted peak
        self._highlighted_fit_result: Optional['FitResult'] = None # Reference to the FitResult whose line is highlighted

        # --- Widget Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Use the entire widget area
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # --- Widget Sizing Policies ---
        # Allow the widget and canvas to expand to fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry() # Ensure initial size calculation

        # --- Final Setup Steps ---
        self._setup_initial_plot_state()    # Configure axes labels, title, grid, annotation
        self._connect_matplotlib_events() # Connect hover and pick handlers
        self.apply_theme_colors()         # Apply initial theme based on config

    def _setup_initial_plot_state(self):
        """Configures the initial appearance of the plot axes and annotation."""
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Spectrum Plot")
        self.ax.grid(True, linestyle=':', alpha=0.6, zorder=-10) # Send grid behind data

        # Initialize the hover annotation object (initially hidden)
        # Using clip_on=True prevents the annotation box from rendering outside the axes area.
        self.annot = self.ax.annotate(
            "", xy=(0,0), xytext=(20,20), # Initial position and offset
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.85), # Rounded box style
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"), # Arrow properties
            visible=False, # Start hidden
            clip_on=True   # Prevent drawing outside axes
        )

    def _connect_matplotlib_events(self):
        """Connects Matplotlib canvas events to handler methods."""
        # Disconnect any existing connections first to prevent duplicates if called again
        if hasattr(self, 'hover_connection_id') and self.hover_connection_id:
            try: self.canvas.mpl_disconnect(self.hover_connection_id)
            except Exception: pass # Ignore errors if already disconnected
        if hasattr(self, 'click_connection_id') and self.click_connection_id:
            try: self.canvas.mpl_disconnect(self.click_connection_id)
            except Exception: pass # Ignore errors if already disconnected

        # Connect motion (hover) and pick (click on designated artists) events
        self.hover_connection_id = self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.click_connection_id = self.canvas.mpl_connect("pick_event", self._on_pick)
        logging.debug("Matplotlib event handlers connected (hover, pick).")

    # --- Artist Management ---
    def _remove_artist(self, key: str):
        """
        Safely removes a plot element (or group) referenced by `key` from the axes
        and updates the internal `_plot_elements` dictionary.
        """
        element = self._plot_elements.get(key)

        try:
            if key == 'fits' and isinstance(element, dict):
                # Special handling for the dictionary of fit lines
                fits_dict: FitLineDict = element
                for line_artist in list(fits_dict.values()): # Iterate over a copy
                    if line_artist:
                        line_artist.remove()
                self._plot_elements['fits'] = {} # Reset to empty dict
                logging.debug("Removed all individual fit lines.")

            elif key == 'nist_lines' and isinstance(element, LineCollection):
                # Special handling for NIST lines (LineCollection) and their annotations (Text)
                element.remove()
                self._plot_elements['nist_lines'] = None
                # Remove associated text annotations
                for ann in self._nist_annotations:
                    ann.remove()
                self._nist_annotations = []
                logging.debug("Removed NIST line collection and annotations.")

            elif isinstance(element, Artist):
                # Handle single, standard artists (Line2D, PathCollection, Legend etc.)
                element.remove()
                self._plot_elements[key] = None
                # logging.debug(f"Removed plot element '{key}'.") # Can be noisy

            # else: element was None or not a removable type, do nothing

        except (AttributeError, ValueError, TypeError) as e:
            # Catch errors indicating the artist might have already been removed or was invalid
            logging.debug(f"Issue removing plot element '{key}' (may already be gone): {e}")
        except Exception as e:
            # Catch unexpected errors during removal
            logging.error(f"Unexpected error removing plot element '{key}': {e}", exc_info=True)

        # Ensure the key is set to None or empty dict after attempting removal
        if key == 'fits':
            self._plot_elements['fits'] = {}
        elif key == 'nist_lines':
            self._plot_elements['nist_lines'] = None
            self._nist_annotations = []
        elif key in self._plot_elements:
             self._plot_elements[key] = None


    def clear_plot(self, redraw: bool = True):
        """
        Clears all plotted data, annotations, and resets axes to initial state.
        Attempts to preserve the current zoom/pan state.
        """
        logging.info("Clearing plot widget.")
        try:
            # --- Preserve Zoom State ---
            # Store current limits only if the axes contain data or have non-default limits
            xlim, ylim = None, None
            if self.ax and (self.ax.has_data() or self.ax.get_xlim() != (0.0, 1.0) or self.ax.get_ylim() != (0.0, 1.0)):
                try:
                     xlim = self.ax.get_xlim()
                     ylim = self.ax.get_ylim()
                     # Basic check for validity (finite and non-inverted)
                     if not (all(np.isfinite(v) for v in xlim + ylim) and xlim[0] < xlim[1] and ylim[0] < ylim[1]):
                          logging.warning(f"Invalid limits detected before clear: X={xlim}, Y={ylim}. Will autoscale.")
                          xlim, ylim = None, None # Invalidate if limits are bad
                except Exception as e:
                     logging.warning(f"Could not get axes limits before clearing: {e}. Will autoscale.")
                     xlim, ylim = None, None

            # --- Remove All Plotted Elements ---
            # Iterate through a copy of keys as the dictionary might change during removal
            for key in list(self._plot_elements.keys()):
                 self._remove_artist(key)

            # --- Clear Axes and Reset State ---
            self.ax.cla() # Clears axes completely (artists, labels, title, etc.)
            self._setup_initial_plot_state() # Reapply title, labels, grid, reset annotation

            # --- Clear Data References ---
            self._peaks_ref = []
            self._matches_ref = []
            self._highlighted_peak_list_index = None
            self._highlighted_fit_result = None

            # --- Restore Zoom State (if preserved and valid) ---
            if xlim is not None and ylim is not None:
                try:
                    self.ax.set_xlim(xlim)
                    self.ax.set_ylim(ylim)
                    logging.debug(f"Restored previous plot limits: X={xlim}, Y={ylim}")
                except ValueError as ve:
                    logging.warning(f"Could not restore previous plot limits after clear: {ve}. Autoscaling.")
                    self.ax.relim()
                    self.ax.autoscale_view(True, True, True)
            else:
                 # Ensure autoscaling if limits weren't preserved
                 self.ax.relim()
                 self.ax.autoscale_view(True, True, True)

            # --- Redraw ---
            if redraw:
                self._redraw_canvas()

        except Exception as e:
            logging.error(f"Error during clear_plot: {e}", exc_info=True)


    # --- Plotting Methods ---

    # Use STRING LITERAL 'Spectrum' for type hint
    def plot_spectrum(self, spectrum: Optional[Spectrum],
                      plot_raw: bool = True, plot_processed: bool = True, plot_baseline: bool = True):
        """ Plots the main spectral data lines (raw, processed, baseline). """
        if not CORE_MODULES_AVAILABLE:
            logging.error("Cannot plot spectrum: Core modules not loaded.")
            return
            
        # --- DETAILED DEBUGGING ---
        logging.debug(f"--- Inside plot_spectrum ---")
        logging.debug(f"Received spectrum object: {spectrum}") # Uses __str__ from Spectrum class
        if spectrum is not None:
             # Check the actual type being received
             logging.debug(f"Type of received spectrum: {type(spectrum)}")
             # Check against the expected type (imported Spectrum)
             is_correct_instance = isinstance(spectrum, Spectrum)
             logging.debug(f"Is instance of imported Spectrum?: {is_correct_instance}")
             # Check attributes explicitly
             has_wl = hasattr(spectrum, 'wavelengths')
             logging.debug(f"Has 'wavelengths' attribute?: {has_wl}")
             if has_wl:
                  wl_is_none = spectrum.wavelengths is None
                  logging.debug(f"'wavelengths' is None?: {wl_is_none}")
                  if not wl_is_none:
                       try:
                           wl_len = len(spectrum.wavelengths)
                           logging.debug(f"Length of 'wavelengths'?: {wl_len}")
                           is_empty = wl_len == 0
                           logging.debug(f"'wavelengths' is empty?: {is_empty}")
                       except TypeError:
                           logging.error(f"'wavelengths' does not support len(): type={type(spectrum.wavelengths)}")
                           is_empty = True # Treat as empty if len fails
                  else: is_empty = True # Treat as empty if None
             else: is_empty = True # Treat as empty if no attribute
        else:
             logging.debug("Received spectrum is None.")
             

    # Use STRING LITERAL 'Peak' for type hint
    def plot_peaks(self, peaks: List['Peak']):
        """
        Plots markers for detected and fitted peaks based on a list of Peak objects.

        Args:
            peaks (List['Peak']): The list of Peak data objects. Old markers are removed.
        """
        if not CORE_MODULES_AVAILABLE:
            logging.error("Cannot plot peaks: Core modules not loaded.")
            return
        if not isinstance(peaks, list):
             logging.error(f"plot_peaks expects a list, but got {type(peaks)}. Ignoring.")
             return

        self._peaks_ref = peaks # Store reference to the provided peak data

        # --- Remove old peak markers and any highlight ---
        self._remove_artist('det')
        self._remove_artist('fit')
        self._remove_artist('highlight')
        self._highlighted_peak_list_index = None

        # --- Prepare Legend Handles ---
        # Get existing legend handles, excluding the peak markers we just removed
        current_legend: Optional[Legend] = self._plot_elements.get('legend') # type: ignore
        handles = []
        if current_legend and hasattr(current_legend, 'legendHandles'):
            handles = [h for h in current_legend.legendHandles if h not in [self._plot_elements.get('det'), self._plot_elements.get('fit')]]

        if not peaks:
            logging.debug("plot_peaks called with empty list. Peak markers cleared.")
            self._update_legend(handles) # Update legend (removes peak entries)
            self._redraw_canvas()
            return

        logging.info(f"Plotting {len(peaks)} peak markers.")

        # --- Get Base Spectrum Line for Y-Coordinates ---
        # Peaks are positioned vertically based on the 'processed' or 'raw' line intensity.
        line_for_y = self._plot_elements.get('proc') or self._plot_elements.get('raw')
        if not isinstance(line_for_y, Line2D):
            logging.warning("Cannot plot peak markers: No 'processed' or 'raw' spectrum line plotted.")
            self._update_legend(handles)
            self._redraw_canvas()
            return

        try:
            plot_wl, plot_int = line_for_y.get_xdata(), line_for_y.get_ydata()
            if len(plot_wl) == 0: # Check if the base line actually has data
                logging.warning("Cannot plot peak markers: Base spectrum line is empty.")
                self._update_legend(handles)
                self._redraw_canvas()
                return
        except Exception as e:
            logging.error(f"Error getting data from base spectrum line for peak plotting: {e}", exc_info=True)
            return

        # --- Prepare Data for Scatter Plots ---
        # Collect coordinates and original list indices for detected vs fitted peaks
        det_x, det_y, det_indices = [], [], []
        fit_x, fit_y, fit_indices = [], [], []

        for i, peak in enumerate(peaks):
            # Validate peak object
            if not isinstance(peak, Peak) or not hasattr(peak, 'wavelength_fitted_or_detected'):
                logging.warning(f"Skipping invalid peak data at index {i}.")
                continue

            marker_wl = peak.wavelength_fitted_or_detected

            # Interpolate Y value safely
            marker_y = np.interp(marker_wl, plot_wl, plot_int, left=np.nan, right=np.nan)

            # Ensure coordinates are valid numbers before adding
            if not (np.isfinite(marker_wl) and np.isfinite(marker_y)):
                logging.warning(f"Skipping peak {i} at λ={marker_wl:.4f}: Invalid coordinates (Y={marker_y:.2f}). Peak might be outside spectrum range.")
                continue

            # Check if the peak has a successful 'best_fit' attribute
            has_fit = hasattr(peak, 'best_fit') and peak.best_fit and getattr(peak.best_fit, 'success', False)

            if has_fit:
                fit_x.append(marker_wl)
                fit_y.append(marker_y)
                fit_indices.append(i) # Store the original list index for this fitted peak
            else:
                det_x.append(marker_wl)
                det_y.append(marker_y)
                det_indices.append(i) # Store the original list index for this detected peak

        # --- Create Scatter Plots ---
        plot_colors = self.config.get('plotting', {})
        det_color = plot_colors.get('peak_detected_color', 'red')
        fit_color = plot_colors.get('peak_fitted_color', 'lime')
        marker_size_det = 35
        marker_size_fit = 50 # Slightly larger for fitted 'o'
        picker_radius = 5 # Pixel radius for enabling 'pick_event'

        try:
            if det_x:
                scatter_det = self.ax.scatter(det_x, det_y, marker="x", color=det_color,
                                              s=marker_size_det, label=f"Detected ({len(det_x)})",
                                              zorder=5, picker=picker_radius)
                # Attach the list of original indices to the artist object for use in _on_pick
                scatter_det.peak_list_indices = det_indices
                self._plot_elements['det'] = scatter_det
                handles.append(scatter_det)

            if fit_x:
                scatter_fit = self.ax.scatter(fit_x, fit_y, marker="o", facecolors='none', edgecolors=fit_color,
                                              s=marker_size_fit, label=f"Fitted ({len(fit_x)})",
                                              zorder=6, picker=picker_radius) # Higher zorder, enable picking
                scatter_fit.peak_list_indices = fit_indices
                self._plot_elements['fit'] = scatter_fit
                handles.append(scatter_fit)

        except Exception as e:
            logging.error(f"Error creating peak scatter plots: {e}", exc_info=True)

        # --- Final Update ---
        self._update_legend(handles)
        self._redraw_canvas()

    # Use STRING LITERALS 'Peak' and 'FitResult' for type hints
    def plot_fit_lines(self, peaks: List['Peak'], highlight_fit: Optional['FitResult'] = None):
        """
        Plots individual fit profile lines for successful fits within the peaks list.

        Args:
            peaks (List['Peak']): List of peaks containing fit results.
            highlight_fit (Optional['FitResult']): A specific FitResult whose line should be visually highlighted.
        """
        if not CORE_MODULES_AVAILABLE:
            logging.error("Cannot plot fit lines: Core modules not loaded.")
            return
        if not isinstance(peaks, list):
             logging.error(f"plot_fit_lines expects a list, but got {type(peaks)}. Ignoring.")
             return

        # --- Clear existing fit lines and store highlight reference ---
        self._remove_artist('fits') # This removes all lines in the 'fits' dictionary
        self._highlighted_fit_result = highlight_fit

        if not peaks:
            logging.debug("plot_fit_lines called with empty peak list.")
            self._redraw_canvas() # Ensure plot is clean if no peaks
            return

        # --- Get Base Spectrum Data (Wavelengths) ---
        base_line = self._plot_elements.get('proc') or self._plot_elements.get('raw')
        if not isinstance(base_line, Line2D):
            logging.warning("Cannot plot fit lines: No base spectrum ('proc' or 'raw') found.")
            return
        try:
            plot_wl = base_line.get_xdata()
            if len(plot_wl) == 0:
                 logging.warning("Cannot plot fit lines: Base spectrum has no wavelength data.")
                 return
        except Exception as e:
             logging.error(f"Error getting wavelength data for fit lines: {e}", exc_info=True)
             return

        # --- Get Baseline Data (Y-values) ---
        baseline_y: Optional[np.ndarray] = None
        baseline_element = self._plot_elements.get('base')
        if isinstance(baseline_element, Line2D):
            try:
                baseline_y_data = baseline_element.get_ydata()
                # Critical check: ensure baseline length matches wavelength array
                if len(baseline_y_data) == len(plot_wl):
                    baseline_y = baseline_y_data
                    logging.debug("Using plotted baseline for fit line Y-offset.")
                else:
                    logging.warning(f"Baseline length ({len(baseline_y_data)}) differs from wavelength length ({len(plot_wl)}). Fit lines will not include baseline offset.")
            except Exception as e:
                logging.error(f"Error getting baseline Y-data: {e}")

        # --- Plot Individual Fit Lines ---
        new_fits_dict: FitLineDict = {}
        plot_colors = self.config.get('plotting', {})
        default_fit_color = plot_colors.get('fit_line_color', 'magenta')
        highlight_color = plot_colors.get('highlight_fit_color', 'yellow')

        num_plotted = 0
        for peak_index, peak in enumerate(peaks):
            if not isinstance(peak, Peak): continue # Skip invalid entries

            # Gather all *successful* fits (best + alternatives) for this peak
            fits_to_plot: Dict[str, 'FitResult'] = {}
            if hasattr(peak, 'alternative_fits') and isinstance(peak.alternative_fits, dict):
                for prof, fit in peak.alternative_fits.items():
                    if isinstance(fit, FitResult) and getattr(fit, 'success', False):
                        fits_to_plot[prof] = fit
            if hasattr(peak, 'best_fit') and isinstance(peak.best_fit, FitResult) and getattr(peak.best_fit, 'success', False):
                profile_type = getattr(peak.best_fit, 'profile_type', 'unknown_best_fit')
                fits_to_plot[profile_type] = peak.best_fit # Overwrites alternative if same type

            if not fits_to_plot: continue # No successful fits found for this peak

            # --- Plot each successful fit for the current peak ---
            for profile_name, fit in fits_to_plot.items():
                 # Basic checks on the FitResult object
                 if not hasattr(fit, 'center') or not hasattr(fit, 'amplitude') or not hasattr(fit, 'width'):
                      logging.warning(f"Skipping fit {profile_name} for peak {peak_index}: Missing essential attributes.")
                      continue
                 if not all(np.isfinite(getattr(fit, attr, np.nan)) for attr in ['center', 'amplitude', 'width']):
                      logging.warning(f"Skipping fit {profile_name} for peak {peak_index}: Non-finite parameters.")
                      continue

                 center = fit.center
                 # --- Determine X-range for plotting this fit ---
                 # Use FWHM if available, else estimate from width, fallback to default
                 fwhm = getattr(fit, 'fwhm', None)
                 width = fit.width
                 plot_range_multiplier = 3.0 # Plot roughly +/- 1.5 * FWHM or equivalent
                 half_plot_width = 0.5 # Default half-width (nm)

                 if fwhm is not None and np.isfinite(fwhm) and fwhm > 1e-6:
                      half_plot_width = fwhm * plot_range_multiplier / 2.0
                 elif width is not None and np.isfinite(width) and width > 1e-9:
                      # Estimate FWHM based on width (depends on profile, use rough factor)
                      # Gauss: FWHM ≈ 2.355*sigma; Lorentz: FWHM = 2*gamma (HWHM)
                      half_plot_width = width * 1.5 * plot_range_multiplier / 2.0
                 else:
                      logging.debug(f"Using default plot width for fit P{peak_index}/{profile_name}")

                 min_wl, max_wl = center - half_plot_width, center + half_plot_width

                 # --- Generate X and Y values for the fit line ---
                 # Select original wavelengths within range, or generate points if too sparse
                 mask = (plot_wl >= min_wl) & (plot_wl <= max_wl)
                 x_fit = plot_wl[mask]
                 num_fit_points = 100 # Points for smooth curve generation
                 if len(x_fit) < 5: # If very few original points fall in range
                      x_fit = np.linspace(min_wl, max_wl, num_fit_points)

                 if len(x_fit) == 0: continue # Skip if range is somehow empty

                 # Generate Y values (relative to baseline = 0)
                 y_fit_relative = self._generate_fit_y(fit, x_fit)
                 if y_fit_relative is None: continue # Failed to generate profile

                 # Add baseline offset if available
                 if baseline_y is not None:
                      try:
                           y_baseline_interp = np.interp(x_fit, plot_wl, baseline_y)
                           y_fit_absolute = y_fit_relative + y_baseline_interp
                      except Exception as interp_e:
                           logging.warning(f"Error interpolating baseline for fit P{peak_index}/{profile_name}: {interp_e}. Plotting relative to zero.")
                           y_fit_absolute = y_fit_relative # Fallback
                 else:
                      y_fit_absolute = y_fit_relative # No baseline to add

                 # --- Determine Style and Plot ---
                 is_highlighted = (highlight_fit is not None and fit == highlight_fit) # Direct object comparison

                 color = highlight_color if is_highlighted else default_fit_color
                 ls = '-' if is_highlighted else '-.'
                 lw = 1.5 if is_highlighted else 0.8
                 alpha = 1.0 if is_highlighted else 0.7
                 zorder = 7 if is_highlighted else 4 # Highlighted above normal fits

                 try:
                      line_key = (peak_index, profile_name) # Use original peak index and profile name as key
                      line, = self.ax.plot(x_fit, y_fit_absolute, color=color, ls=ls, lw=lw,
                                           alpha=alpha, zorder=zorder, label='_nolegend_') # No legend entries
                      # Store reference to the FitResult object on the line itself (useful for hover/pick later if needed)
                      line.fit_result_ref = fit
                      # Store peak index on fit object if it doesn't have one (sometimes useful)
                      if not hasattr(fit, 'peak_index'): fit.peak_index = peak_index

                      new_fits_dict[line_key] = line
                      num_plotted += 1
                 except Exception as plot_e:
                      logging.error(f"Error plotting fit line P{peak_index}/{profile_name}: {plot_e}", exc_info=True)

        # --- Update internal state and redraw ---
        self._plot_elements['fits'] = new_fits_dict
        if num_plotted > 0:
             logging.info(f"Plotted {num_plotted} individual fit lines.")
        if highlight_fit:
             logging.info(f"Highlighted fit: Peak {getattr(highlight_fit, 'peak_index', 'N/A')} / {getattr(highlight_fit, 'profile_type', 'N/A')}")

        self._redraw_canvas()

    # Use STRING LITERAL 'FitResult' for type hint
    def highlight_fit_line(self, fit_result: Optional['FitResult']):
        """
        Highlights a specific fit line by replotting all fits with new styling.

        Args:
            fit_result (Optional['FitResult']): The fit to highlight. Pass None to clear highlight.
        """
        if not CORE_MODULES_AVAILABLE:
             logging.error("Cannot highlight fit line: Core modules not loaded.")
             return

        # Validate input type (runtime check)
        if fit_result is not None and not isinstance(fit_result, FitResult):
             logging.warning(f"highlight_fit_line called with invalid type: {type(fit_result)}. Clearing highlight.")
             fit_result = None # Treat invalid input as clearing highlight

        current_highlight = self._highlighted_fit_result

        # Only replot if the requested highlight state *changes*
        if fit_result != current_highlight:
            logging.info(f"Changing fit highlight to: {fit_result if fit_result else 'None'}")
            # plot_fit_lines handles removal of old lines and plotting with the new highlight state
            self.plot_fit_lines(self._peaks_ref, highlight_fit=fit_result)
        else:
            logging.debug(f"Fit highlight requested ({fit_result}) is same as current. No change.")


    # Use STRING LITERAL 'NISTMatch' for type hint
    def plot_nist_matches(self, matches: List['NISTMatch'], clear_previous: bool = True):
        """
        Plots vertical lines and labels for NIST database matches.

        Args:
            matches (List['NISTMatch']): List of NISTMatch objects to plot.
            clear_previous (bool): If True, remove previously plotted NIST lines/labels first.
        """
        if not CORE_MODULES_AVAILABLE:
             logging.error("Cannot plot NIST matches: Core modules not loaded.")
             return
        if not isinstance(matches, list):
             logging.error(f"plot_nist_matches expects a list, got {type(matches)}. Ignoring.")
             return

        if clear_previous:
            self.clear_nist_matches() # Clears lines and annotations

        if not matches:
            logging.debug("plot_nist_matches called with empty list.")
            self._redraw_canvas() # Redraw needed if clear_previous was True
            return

        self._matches_ref = matches # Store reference to the match data
        logging.info(f"Plotting {len(matches)} NIST matches.")

        # --- Prepare Data Containers ---
        lines_segments = [] # List of [(x0, y0), (x1, y1)] for LineCollection
        lines_colors = []   # List of colors for each segment
        new_nist_annotations: List[MplText] = [] # Store newly created Text objects

        # --- Determine Y-Range for Lines and Labels ---
        try:
            current_ylim = self.ax.get_ylim()
            # Handle edge case where plot might be empty or has default (0,1) limits
            if current_ylim == (0.0, 1.0) and not self.ax.has_data():
                # Attempt to guess a reasonable Y max based on plotted data (if any remains)
                y_max_guess = 1.0
                for key in ['proc', 'raw']:
                    line = self._plot_elements.get(key)
                    if isinstance(line, Line2D) and len(line.get_ydata()) > 0:
                        try: y_max_guess = max(y_max_guess, np.nanmax(line.get_ydata()))
                        except ValueError: pass # Ignore if data contains only NaNs
                current_ylim = (self.ax.dataLim.y0 if self.ax.dataLim.y0 < y_max_guess else 0, y_max_guess * 1.1)

            # Calculate positions relative to current Y limits
            yrange = current_ylim[1] - current_ylim[0]
            if yrange <= 0: yrange = 1.0 # Avoid division by zero or negative range
            line_y_start = current_ylim[0]
            # Position lines and labels towards the top, adjust factors as needed
            line_y_end = current_ylim[0] + yrange * 0.85
            label_y_pos = current_ylim[0] + yrange * 0.90
        except Exception as e:
            logging.error(f"Could not determine Y limits for NIST lines: {e}. Using relative defaults.")
            # Fallback to relative positioning if limits fail
            line_y_start, line_y_end, label_y_pos = 0.0, 0.85, 0.90
            # Note: These might not render well if axes limits change later. Absolute calculation is preferred.

        # --- Assign Colors Based on Element ---
        unique_elements = sorted(list(set(getattr(m, 'element', None) for m in matches if hasattr(m, 'element') and m.element)))
        color_map = {}
        default_color = 'grey'
        if unique_elements:
             try:
                  cmap = plt.get_cmap('tab10') # Standard qualitative colormap
                  num_colors = min(cmap.N, 10) # Use at most 10 distinct colors from tab10
                  color_map = {elem: cmap(i % num_colors) for i, elem in enumerate(unique_elements)}
             except Exception as e:
                  logging.warning(f"Could not get colormap for NIST elements: {e}. Using default color.")

        # --- Create Line Segments and Text Annotations ---
        num_plotted = 0
        for match in matches:
            # Validate match object and wavelength
            if not isinstance(match, NISTMatch) or not hasattr(match, 'wavelength_db'): continue
            wl_db = match.wavelength_db
            if not np.isfinite(wl_db): continue # Skip matches with invalid wavelength

            # Define line segment
            lines_segments.append([(wl_db, line_y_start), (wl_db, line_y_end)])

            # Determine color and label
            element = getattr(match, 'element', None)
            color = color_map.get(element, default_color) if element else default_color
            lines_colors.append(color)
            ion_state_str = getattr(match, 'ion_state_str', '')
            # Combine element and ion state, handle missing element
            label = f"{element} {ion_state_str}".strip() if element else f"? {ion_state_str}".strip()

            # Create text annotation
            try:
                txt = self.ax.text(wl_db, label_y_pos, label, rotation=90,
                                   ha='center', va='bottom', # Align bottom of text at label_y_pos
                                   fontsize=7, color=color, clip_on=True) # clip_on keeps labels in view
                new_nist_annotations.append(txt)
                num_plotted += 1
            except Exception as text_e:
                logging.error(f"Failed to create NIST text annotation for '{label}' at {wl_db:.2f}: {text_e}")

        # --- Plot using LineCollection for Efficiency ---
        if lines_segments:
            try:
                line_collection = LineCollection(lines_segments, colors=lines_colors,
                                                 linewidths=0.7, alpha=0.8,
                                                 label='_nolegend_', zorder=0) # Plot behind data
                self.ax.add_collection(line_collection)
                self._plot_elements['nist_lines'] = line_collection
                self._nist_annotations = new_nist_annotations # Store references to the text objects
                logging.debug(f"Added {num_plotted} NIST lines/annotations via LineCollection.")
            except Exception as e:
                logging.error(f"Failed to add NIST LineCollection to plot: {e}", exc_info=True)
                # Clean up any text annotations created if LineCollection failed
                for ann in new_nist_annotations: ann.remove()
                self._nist_annotations = []
        else:
             logging.debug("No valid NIST match data resulted in plottable lines.")

        self._redraw_canvas()


    def clear_nist_matches(self):
        """Removes NIST match lines and their associated text labels."""
        logging.debug("Clearing NIST matches from plot.")
        # _remove_artist handles removing both the LineCollection and the Text annotations
        self._remove_artist('nist_lines')
        self._matches_ref = [] # Clear data reference
        # No redraw needed here, _remove_artist doesn't redraw. Caller should handle redraw if needed immediately.


    def highlight_peak(self, peak_list_index: Optional[int]):
        """
        Highlights a specific peak marker using its index in the original list.

        Args:
            peak_list_index (Optional[int]): Index in the list given to `plot_peaks`.
                                             Pass None to remove the highlight.
        """
        # --- Remove Previous Highlight ---
        self._remove_artist('highlight') # Removes the old highlight scatter plot, if any
        self._highlighted_peak_list_index = peak_list_index # Update stored index

        # --- Input Validation ---
        if peak_list_index is None:
            logging.debug("Clearing peak highlight.")
            self._redraw_canvas() # Redraw needed to ensure highlight is gone
            return

        if not isinstance(self._peaks_ref, list) or not (0 <= peak_list_index < len(self._peaks_ref)):
            logging.warning(f"Cannot highlight peak: Index {peak_list_index} is invalid for current peak list (size {len(self._peaks_ref)}).")
            self._redraw_canvas() # Redraw if index was bad but highlight was removed
            return

        peak = self._peaks_ref[peak_list_index]
        if not isinstance(peak, Peak) or not hasattr(peak, 'wavelength_fitted_or_detected'):
            logging.warning(f"Cannot highlight peak index {peak_list_index}: Invalid Peak object or missing wavelength.")
            self._redraw_canvas()
            return

        # --- Determine Highlight Coordinates ---
        highlight_x = peak.wavelength_fitted_or_detected
        # Use the same spectrum line ('proc' or 'raw') that was used for plotting peak markers
        line_for_y = self._plot_elements.get('proc') or self._plot_elements.get('raw')

        if not isinstance(line_for_y, Line2D):
            logging.warning(f"Cannot determine highlight Y-coordinate for peak index {peak_list_index}: Base spectrum line not found.")
            self._redraw_canvas()
            return

        highlight_y = np.nan # Default to NaN
        try:
            wl_data, int_data = line_for_y.get_xdata(), line_for_y.get_ydata()
            if len(wl_data) > 0: # Ensure data exists for interpolation
                 highlight_y = np.interp(highlight_x, wl_data, int_data, left=np.nan, right=np.nan)
        except Exception as e:
            logging.error(f"Error interpolating highlight Y-coordinate: {e}", exc_info=True)
            # highlight_y remains NaN

        # --- Plot Highlight Marker ---
        if np.isfinite(highlight_x) and np.isfinite(highlight_y):
            plot_colors = self.config.get('plotting', {})
            highlight_color = plot_colors.get('highlight_peak_color', 'yellow')
            marker_size = 150 # Larger marker for visibility
            line_width = 1.5
            try:
                # Use scatter for the highlight marker (a single point)
                highlight_scatter = self.ax.scatter(
                    [highlight_x], [highlight_y], marker='o', s=marker_size,
                    facecolors='none', edgecolors=highlight_color,
                    lw=line_width, zorder=10, label='_nolegend_' # High z-order, no legend entry
                )
                self._plot_elements['highlight'] = highlight_scatter
                logging.info(f"Highlighted peak list index {peak_list_index} at ({highlight_x:.4f}, {highlight_y:.2f})")
            except Exception as e:
                logging.error(f"Error plotting highlight marker: {e}", exc_info=True)
        else:
            logging.warning(f"Cannot highlight peak index {peak_list_index}: Invalid/NaN coordinates (X={highlight_x}, Y={highlight_y}).")

        self._redraw_canvas()


    # --- Helper Methods ---

    # Use STRING LITERAL 'FitResult' for type hint
    def _generate_fit_y(self, fit: 'FitResult', x: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculates the Y values for a given FitResult object over an array of X values.

        Args:
            fit ('FitResult'): The fit result containing profile type and parameters.
            x (np.ndarray): The wavelength values to calculate the fit profile for.

        Returns:
            Optional[np.ndarray]: The calculated Y values, or None if calculation fails
                                  (e.g., missing functions, invalid parameters).
        """
        if not CORE_MODULES_AVAILABLE:
            # Log error only once per instance or less frequently if needed
            if not hasattr(self, '_logged_missing_core_func_error'):
                 logging.error("Cannot generate fit Y values: Core processing functions (gaussian, etc.) are not available.")
                 self._logged_missing_core_func_error = True
            return None
        if not isinstance(fit, FitResult):
             logging.warning("Cannot generate fit Y: Invalid FitResult object provided.")
             return None

        try:
            # Extract parameters safely using getattr with defaults
            amp = getattr(fit, 'amplitude', np.nan)
            cen = getattr(fit, 'center', np.nan)
            wid = getattr(fit, 'width', np.nan) # Note: 'width' interpretation depends on profile
            prof = getattr(fit, 'profile_type', None)

            # --- Parameter Validation ---
            if not prof:
                 logging.warning(f"Cannot generate fit Y: Missing profile type in FitResult.")
                 return None
            if not (np.isfinite(amp) and np.isfinite(cen)):
                 logging.warning(f"Cannot generate fit Y for '{prof}': Invalid amplitude ({amp}) or center ({cen}).")
                 return None

            # --- Call Appropriate Fitting Function ---
            if prof == 'Gaussian':
                if not np.isfinite(wid) or wid <= 0:
                    logging.warning(f"Invalid width (sigma={wid}) for Gaussian fit.")
                    return None
                return gaussian(x, amp, cen, wid)

            elif prof == 'Lorentzian':
                # Lorentzian width parameter often 'gamma' (HWHM), ensure it's positive
                if not np.isfinite(wid) or wid <= 0:
                    logging.warning(f"Invalid width (gamma={wid}) for Lorentzian fit.")
                    return None
                return lorentzian(x, amp, cen, wid)

            elif prof == 'PseudoVoigt':
                # Requires FWHM and eta (mixing parameter) in addition to amp, cen
                eta = getattr(fit, 'mixing_param_eta', np.nan)
                fwhm = getattr(fit, 'fwhm', np.nan)
                if not (np.isfinite(eta) and np.isfinite(fwhm)) or fwhm <= 0:
                    logging.warning(f"Invalid parameters for PseudoVoigt (FWHM={fwhm}, eta={eta}).")
                    return None
                # Ensure eta is within [0, 1] range expected by pseudo_voigt function
                eta_clipped = np.clip(eta, 0.0, 1.0)
                if eta != eta_clipped:
                    logging.debug(f"Clipped PseudoVoigt eta parameter from {eta} to {eta_clipped}.")
                return pseudo_voigt(x, amp, cen, fwhm, eta_clipped)

            else:
                logging.warning(f"Cannot generate fit Y: Unknown profile type '{prof}'.")
                return None

        except Exception as e:
            logging.error(f"Error generating Y-values for fit profile '{getattr(fit, 'profile_type', 'N/A')}': {e}", exc_info=True)
            return None

    def _update_legend(self, specific_handles: Optional[List[Artist]] = None):
        """
        Refreshes the plot legend, ensuring no duplicate entries and applying theme.

        Args:
            specific_handles: If provided, only these handles (and their labels) are
                              considered for the legend. Otherwise, uses handles from axes.
        """
        try:
            # --- Gather Handles and Labels ---
            handles_to_consider: List[Artist] = []
            labels_to_consider: List[str] = []

            if specific_handles is not None:
                 handles_to_consider = specific_handles
                 labels_to_consider = [getattr(h, 'get_label', lambda: '')() for h in handles_to_consider]
            else:
                 # Get all handles/labels currently associated with the axes
                 handles_to_consider, labels_to_consider = self.ax.get_legend_handles_labels()

            # --- Filter and Deduplicate ---
            # Keep only handles with valid, non-private labels. Use a dict to automatically handle duplicates (last one wins).
            valid_legend_items: Dict[str, Artist] = {}
            for handle, label in zip(handles_to_consider, labels_to_consider):
                if handle and label and not label.startswith('_'):
                    valid_legend_items[label] = handle

            # --- Remove Old Legend ---
            old_legend = self._plot_elements.get('legend')
            if isinstance(old_legend, Legend):
                try:
                    old_legend.remove()
                except Exception as e:
                    logging.debug(f"Issue removing previous legend: {e}")
            self._plot_elements['legend'] = None

            # --- Create New Legend ---
            if valid_legend_items:
                legend_handles = list(valid_legend_items.values())
                legend_labels = list(valid_legend_items.keys())
                new_legend = self.ax.legend(legend_handles, legend_labels, fontsize='small', loc='best')
                self._plot_elements['legend'] = new_legend
                self._apply_legend_theme() # Apply theme colors immediately
                logging.debug(f"Legend updated with labels: {legend_labels}")
            else:
                logging.debug("No valid items for legend.")

        except Exception as e:
            logging.error(f"Failed to update plot legend: {e}", exc_info=True)

    def _apply_legend_theme(self):
        """Applies theme colors (background, text) to the current legend frame and text."""
        legend = self._plot_elements.get('legend')
        if not isinstance(legend, Legend): return

        try:
            # Determine theme (use rcParams as primary source, config as override/fallback)
            is_dark = plt.rcParams.get('figure.facecolor', '#FFFFFF').lower() < '#808080' # Guess based on bg color
            style_cfg = self.config.get('style', {})
            theme_name = style_cfg.get('default_theme', None)
            if theme_name: is_dark = 'dark' in theme_name.lower()

            # Get colors: Prioritize rcParams, fallback to simple dark/light defaults
            default_text = 'white' if is_dark else 'black'
            default_bg = '#333333' if is_dark else '#FFFFFF'
            default_edge = '#555555' if is_dark else '#CCCCCC'

            text_color = plt.rcParams.get('legend.labelcolor', plt.rcParams.get('text.color', default_text))
            bg_color_str = plt.rcParams.get('legend.facecolor', plt.rcParams.get('axes.facecolor', default_bg))
            edge_color_str = plt.rcParams.get('legend.edgecolor', plt.rcParams.get('grid.color', default_edge))

            # Apply colors
            frame = legend.get_frame()
            bg_color_rgba = matplotlib.colors.to_rgba(bg_color_str, alpha=0.85) # Semi-transparent background
            frame.set(facecolor=bg_color_rgba, edgecolor=edge_color_str, linewidth=0.5)
            for text in legend.get_texts():
                text.set_color(text_color)

        except Exception as e:
            logging.warning(f"Could not apply theme settings to legend: {e}", exc_info=True)


    def _update_annotation(self, target_element: Artist, data_info: Dict):
        """
        Updates the text and position of the hover annotation box.

        Args:
            target_element (Artist): The Matplotlib artist being hovered over.
            data_info (Dict): Dictionary containing info like 'x', 'y', 'peak_info'.
        """
        try:
            # --- Set Annotation Position ---
            # Use exact data point coordinates from data_info
            pos = (data_info.get('x', 0), data_info.get('y', 0))
            self.annot.xy = pos
            text = "" # Initialize annotation text

            # --- Format Text Based on Element Type ---
            if isinstance(target_element, Line2D):
                # --- Line Hover (Spectrum, Baseline, Fit Line) ---
                label = getattr(target_element, 'get_label', lambda: '')().lower()
                prefix = "Data" # Default prefix
                if label.startswith("raw"): prefix = "Raw"
                elif label.startswith("proc"): prefix = "Processed"
                elif label.startswith("base"): prefix = "Baseline"
                elif hasattr(target_element, 'fit_result_ref'): # Check if it's one of our fit lines
                     fit_ref = getattr(target_element, 'fit_result_ref')
                     profile = getattr(fit_ref,'profile_type', 'Unknown') if fit_ref else 'Unknown'
                     prefix = f"Fit ({profile})"
                elif label and not label.startswith('_'): # Use label if available and public
                     prefix = label.capitalize()

                text = f"{prefix}\nλ: {pos[0]:.4f}\nI: {pos[1]:.2f}"

            elif isinstance(target_element, PathCollection):
                # --- Scatter Hover (Peak Markers) ---
                label = getattr(target_element, 'get_label', lambda: '')().lower()
                prefix = "Peak" # Default
                if label.startswith("det"): prefix = "Detected Peak"
                elif label.startswith("fit"): prefix = "Fitted Peak"
                elif label and '(' in label: prefix = label.split('(')[0].strip().capitalize() # Extract from "Fitted (N)"

                # Attempt to retrieve detailed peak info stored during plotting
                peak_info = data_info.get('peak_info') # This should be a dict from Peak.to_dataframe_row()
                if isinstance(peak_info, dict):
                    # Extract relevant info with defaults
                    det_wav = peak_info.get('Detected Wavelength (nm)', np.nan)
                    fit_cen = peak_info.get('Fitted Center (nm)', np.nan)
                    fit_fwhm = peak_info.get('Fitted FWHM (nm)', np.nan)
                    proc_int = peak_info.get('Processed Intensity', np.nan) # Intensity at detected wav
                    peak_height = peak_info.get('Fitted Amplitude', np.nan) # Fitted amplitude
                    prof = peak_info.get('Fit Profile', '')

                    text = f"{prefix}"
                    # Show detected wavelength if available, else hover wavelength
                    text += f"\nλ Detect: {det_wav:.4f}" if np.isfinite(det_wav) else f"\nλ Hover: {pos[0]:.4f}"
                    # Show intensity at detection point, else hover intensity
                    text += f"\nI Detect: {proc_int:.2f}" if np.isfinite(proc_int) else f"\nI Hover: {pos[1]:.2f}"

                    # Add fit info if available and valid
                    if prof and isinstance(prof, str) and prof != '':
                        text += f"\nFit: {prof}"
                        if np.isfinite(fit_cen): text += f"\n Fit λ: {fit_cen:.4f}"
                        if np.isfinite(peak_height): text += f"\n Fit Amp: {peak_height:.2f}"
                        if np.isfinite(fit_fwhm): text += f"\n Fit FWHM: {fit_fwhm:.4f}"
                else:
                    # Fallback if detailed peak_info wasn't found
                    text = f"{prefix}\nλ: {pos[0]:.4f}\nI: {pos[1]:.2f}"

            # --- Apply Text and Make Visible ---
            if text:
                 self.annot.set_text(text)
                 self.annot.set_visible(True)
            else:
                 # Hide annotation if no text could be generated
                 self.annot.set_visible(False)

        except Exception as e:
            logging.error(f"Error updating annotation: {e}", exc_info=True)
            self.annot.set_visible(False) # Hide annotation on error


    # --- Event Handlers ---

    def _on_hover(self, event):
        """Handles mouse motion events over the canvas for hover effects."""
        # Check if the mouse event occurred within the plot axes
        if not event.inaxes == self.ax:
            # If mouse is outside axes, hide annotation if it's visible
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self._redraw_canvas() # Redraw needed to hide annotation
            return

        # --- Find Hovered Element ---
        # Prioritize scatter points (peaks) over lines
        target_element: Optional[Artist] = None
        info: Dict[str, Any] = {} # Dictionary to store data about the hovered point
        min_dist_sq = float('inf') # Used for finding closest point on lines

        # 1. Check Peak Markers (Scatter Plots)
        # Combine detected and fitted peak scatter artists for checking
        peak_scatters = [s for s in [self._plot_elements.get('det'), self._plot_elements.get('fit')] if isinstance(s, PathCollection)]
        for scatter in peak_scatters:
            contains, ind_dict = scatter.contains(event) # Check if event is within picker radius
            if contains:
                scatter_indices = ind_dict['ind']
                if len(scatter_indices) > 0:
                    # Find the closest point within the picked indices (usually just one)
                    offsets = scatter.get_offsets() # Get all (x, y) coordinates of the scatter plot
                    points_in_radius = offsets[scatter_indices]
                    distances = np.sum((points_in_radius - [event.xdata, event.ydata])**2, axis=1)
                    closest_idx_in_subset = np.argmin(distances)
                    # Map back to the original index in the scatter data array
                    scatter_index = scatter_indices[closest_idx_in_subset]

                    if scatter_index < len(offsets):
                        target_element = scatter
                        pos = offsets[scatter_index]
                        info = {'x': pos[0], 'y': pos[1]}

                        # Retrieve the original peak list index stored on the artist
                        peak_list_indices = getattr(scatter, 'peak_list_indices', None)
                        if peak_list_indices and isinstance(peak_list_indices, list) and scatter_index < len(peak_list_indices):
                            original_list_index = peak_list_indices[scatter_index]
                            # Get detailed info from the referenced Peak object
                            if 0 <= original_list_index < len(self._peaks_ref):
                                peak_obj = self._peaks_ref[original_list_index]
                                if peak_obj and hasattr(peak_obj, 'to_dataframe_row'):
                                    # Use a method on the Peak object to get hover info
                                    info['peak_info'] = peak_obj.to_dataframe_row()
                                else: info['peak_info'] = {'Index': original_list_index} # Basic fallback
                        break # Found a scatter point, stop searching
        # 2. Check Lines (Spectrum, Baseline, Fit Lines) if no peak marker found
        if target_element is None:
            lines_to_check: List[Line2D] = []
            # Add main spectrum/baseline lines
            for key in ['proc', 'raw', 'base']:
                 line = self._plot_elements.get(key)
                 if isinstance(line, Line2D): lines_to_check.append(line)
            # Add individual fit lines
            fits_dict = self._plot_elements.get('fits')
            if isinstance(fits_dict, dict):
                 lines_to_check.extend(l for l in fits_dict.values() if isinstance(l, Line2D))

            for line in lines_to_check:
                 # Use Line2D.contains to check proximity, radius increases tolerance
                 contains, ind_dict = line.contains(event, radius=5)
                 if contains:
                     x_data, y_data = line.get_data()
                     indices = ind_dict['ind'] # Indices of data points near the event
                     if len(indices) > 0:
                          # Find the closest point among the nearby indices
                          distances_sq = (x_data[indices] - event.xdata)**2 + (y_data[indices] - event.ydata)**2
                          closest_local_idx = np.argmin(distances_sq)
                          closest_dist_sq = distances_sq[closest_local_idx]

                          # If this point is closer than any found on other lines, update target
                          if closest_dist_sq < min_dist_sq:
                               min_dist_sq = closest_dist_sq
                               target_element = line
                               global_index = indices[closest_local_idx] # Index in the full x_data/y_data
                               info = {'x': x_data[global_index], 'y': y_data[global_index]}

        # --- Update or Hide Annotation ---
        if target_element:
            # If an element is found, update and show the annotation
            self._update_annotation(target_element, info)
            self._redraw_canvas() # Redraw needed to show/update annotation
        elif self.annot.get_visible():
            # If no element is hovered but annotation is visible, hide it
            self.annot.set_visible(False)
            self._redraw_canvas() # Redraw needed to hide annotation


    def _on_pick(self, event):
        """Handles pick events (typically clicks) on artists with 'picker' enabled."""
        artist = event.artist
        indices = event.ind # Indices of the picked data points within the artist's data

        # Check if the picked artist is one of our peak scatter plots
        if isinstance(artist, PathCollection) and hasattr(artist, 'peak_list_indices') and indices:
            # Get the index of the *first* picked point (usually only one is picked)
            scatter_index = indices[0]
            peak_list_indices = artist.peak_list_indices # Retrieve the stored list of original indices

            if peak_list_indices and isinstance(peak_list_indices, list) and scatter_index < len(peak_list_indices):
                # Map the scatter data index back to the original peak list index
                original_list_index = peak_list_indices[scatter_index]
                logging.info(f"Peak clicked: Scatter Index={scatter_index}, Original Peak List Index={original_list_index}")
                # Emit the signal with the original list index
                self.peak_clicked.emit(original_list_index)
            else:
                logging.warning(f"Picked scatter point (index {scatter_index}), but could not map it back to an original peak list index. Artist: {artist.get_label()}")
        # Can add handling for picking other artist types here if needed later


    # --- Theming and Redrawing ---

    def apply_theme_colors(self, config: Optional[Dict] = None):
        """
        Applies color theme settings from the configuration to plot elements.
        Uses Matplotlib rcParams as fallbacks if config keys are missing.

        Args:
            config (Optional[Dict]): The configuration dictionary. If None, uses self.config.
        """
        if config is None: config = self.config
        if not config:
             logging.debug("apply_theme_colors: No configuration provided, using rcParams defaults.")
             # Rely on Matplotlib's current style or defaults
        else:
             logging.debug("Applying theme colors from configuration.")

        try:
            # Determine base theme (dark/light) - prioritize config, fallback to guessing from rcParams
            style_cfg = config.get('style', {})
            plotting_cfg = config.get('plotting', {}) # Use dedicated plotting section if available
            if not plotting_cfg and 'plotting' in style_cfg: plotting_cfg = style_cfg.get('plotting', {}) # Check within style too

            theme_name = style_cfg.get('default_theme', None)
            # Guess theme if not explicitly set in config by looking at default figure background
            if theme_name:
                 is_dark = 'dark' in theme_name.lower()
            else:
                 # Check rcParams for background color to guess theme mode
                 default_bg = plt.rcParams.get('figure.facecolor', '#FFFFFF')
                 is_dark = matplotlib.colors.to_rgb(default_bg)[0] < 0.5 # Simple brightness check

            logging.debug(f"Theme detected as {'dark' if is_dark else 'light'}.")

            # --- Get Colors (Config > rcParams > Hardcoded Defaults) ---
            def get_color(key: str, default_light: str, default_dark: str) -> str:
                 config_val = plotting_cfg.get(key)
                 if config_val: return config_val
                 # Construct rcParam keys (heuristic)
                 rc_keys = [f'cosmic.{key}', key.replace('_','.')] # Example potential keys
                 for rc_key in rc_keys:
                      rc_val = plt.rcParams.get(rc_key)
                      if rc_val: return rc_val # type: ignore
                 # Fallback based on detected theme mode
                 return default_dark if is_dark else default_light

            bg_color = get_color('background_color', '#FFFFFF', '#2E2E2E')
            text_color = get_color('text_color', '#000000', '#FFFFFF')
            grid_color = get_color('grid_color', '#CCCCCC', '#444444')
            raw_c = get_color('raw_data_color', '#AAAAAA', '#888888') # Slightly darker gray for dark
            proc_c = get_color('processed_data_color', '#0000FF', '#00FFFF') # Blue (light), Cyan (dark)
            base_c = get_color('baseline_color', '#FFA500', '#FF8C00') # Orange (light), DarkOrange (dark)
            det_c = get_color('peak_detected_color', '#FF0000', '#FF4500') # Red (light), OrangeRed (dark)
            fit_c = get_color('peak_fitted_color', '#00FF00', '#ADFF2F') # Lime (light), GreenYellow (dark)
            fit_line_c = get_color('fit_line_color', '#FF00FF', '#DA70D6') # Magenta (light), Orchid (dark)
            hl_peak_c = get_color('highlight_peak_color', '#FFFF00', '#FFFF00') # Yellow (both)
            hl_fit_c = get_color('highlight_fit_color', '#FFFF00', '#FFFFE0') # Yellow (light), LightYellow (dark)
            annot_bg_c = get_color('annotation_background_color', '#FFFFE0', '#3C3F41') # LightYellow, Dark Gray
            annot_text_c = get_color('annotation_text_color', '#000000', '#FFFFFF')

            # --- Apply Colors to Figure, Axes, Grid, Ticks, Labels, Title ---
            self.figure.set_facecolor(bg_color)
            self.ax.set_facecolor(bg_color)
            for spine in self.ax.spines.values(): spine.set_color(text_color)
            self.ax.xaxis.label.set_color(text_color)
            self.ax.yaxis.label.set_color(text_color)
            self.ax.tick_params(axis='x', colors=text_color)
            self.ax.tick_params(axis='y', colors=text_color)
            self.ax.title.set_color(text_color)
            self.ax.grid(True, color=grid_color, linestyle=':', alpha=0.6)

            # --- Apply Colors to Plotted Data Elements ---
            elements = self._plot_elements
            if isinstance(elements.get('raw'), Line2D): elements['raw'].set_color(raw_c) # type: ignore
            if isinstance(elements.get('proc'), Line2D): elements['proc'].set_color(proc_c) # type: ignore
            if isinstance(elements.get('base'), Line2D): elements['base'].set_color(base_c) # type: ignore
            if isinstance(elements.get('det'), PathCollection): elements['det'].set_color(det_c) # type: ignore
            if isinstance(elements.get('fit'), PathCollection): elements['fit'].set_edgecolor(fit_c); elements['fit'].set_facecolor('none') # type: ignore
            if isinstance(elements.get('highlight'), PathCollection): elements['highlight'].set_edgecolor(hl_peak_c) # type: ignore

            # Apply colors to individual fit lines, considering highlight state
            fits_dict = elements.get('fits')
            if isinstance(fits_dict, dict):
                 highlight_ref = self._highlighted_fit_result
                 for line_key, line in fits_dict.items():
                     # Check if this line corresponds to the highlighted fit result
                     is_highlighted = (highlight_ref is not None and hasattr(line, 'fit_result_ref') and line.fit_result_ref == highlight_ref) # Added hasattr check
                     line.set_color(hl_fit_c if is_highlighted else fit_line_c)
                     # Adjust other properties for highlight if needed (already done in plot_fit_lines)
                     # line.set_linestyle('-' if is_highlighted else '-.')
                     # line.set_linewidth(1.5 if is_highlighted else 0.8)
                     # line.set_alpha(1.0 if is_highlighted else 0.7)
                     # line.set_zorder(7 if is_highlighted else 4)

            # --- Apply Colors to Legend and Annotation ---
            self._apply_legend_theme() # Re-apply theme to legend (uses rcParams mostly)

            if self.annot:
                 self.annot.get_bbox_patch().set(facecolor=annot_bg_c, edgecolor=annot_text_c, alpha=0.85)
                 self.annot.set_color(annot_text_c)
                 # Update arrow color if possible (depends on matplotlib version/backend)
                 if hasattr(self.annot, 'arrow_patch') and self.annot.arrow_patch:
                      self.annot.arrow_patch.set_color(annot_text_c)

            # --- Redraw Canvas ---
            self._redraw_canvas()
            logging.debug("Theme colors applied.")

        except Exception as e:
            logging.error(f"Error applying theme colors to plot: {e}", exc_info=True)


    def _redraw_canvas(self):
        """Requests an idle redraw of the Matplotlib canvas."""
        try:
            # draw_idle() is preferred in Qt event loops as it schedules a draw
            # instead of forcing an immediate one, preventing potential GUI freezes.
            self.canvas.draw_idle()
        except Exception as e:
            # Catch errors that might occur during the draw process (e.g., invalid state)
            logging.error(f"Error occurred during canvas redraw: {e}", exc_info=True)

# --- END OF REFACTORED FILE libs_cosmic_forge/ui/views/plot_widget.py ---