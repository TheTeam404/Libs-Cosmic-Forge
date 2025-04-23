# --- START OF MODIFIED FILE libs_cosmic_forge/ui/views/control_panel_view.py ---
"""
Control Panel View for Spectrum Processing settings (Baseline, Denoising, Smoothing).
Uses CollapsibleBox widgets for organization.
"""
import logging
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QPushButton, QLabel, QHBoxLayout, QComboBox,
                             QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QMessageBox, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt, QVariant, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QBrush , QAction, QIcon

# Attempt to import necessary components, provide clear errors if missing
try:
    from ui.widgets.info_button import InfoButton
    from ui.widgets.collapsible_box import CollapsibleBox
    # No core data models needed directly in this panel anymore
except ImportError as e:
    logging.critical(f"Failed to import UI components: {e}. Check project structure and dependencies.")
    raise # Re-raise as these are critical for the UI panel

# Renamed class from previous context (fitting) to general processing
class ProcessingControlPanel(QWidget):
    """Control panel for spectrum processing: Baseline, Denoising, Smoothing."""
    # Signal emitted to trigger processing with current settings
    process_triggered = pyqtSignal(dict)

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Get the main processing section, or an empty dict if it doesn't exist
        self.processing_config = config.get('processing', {})
        self.config = self.processing_config # Keep self.config for compatibility if needed
        self._dark_mode: bool = False # Track theme if needed for styling

        # --- Add attributes to store layout references --- ## ADDED ##
        self.baseline_layout: Optional[QFormLayout] = None
        self.denoising_layout: Optional[QFormLayout] = None
        self.smoothing_layout: Optional[QFormLayout] = None
        self.wavelet_params_widget: Optional[QWidget] = None # Keep reference to container
        self.savitzky_params_widget: Optional[QWidget] = None # Keep reference to container
        # Keep references to the row widgets for setRowVisible ## ADDED ##
        self.poly_order_row_widget: Optional[QWidget] = None
        self.snip_iter_row_widget: Optional[QWidget] = None

        # --- List of available wavelets (can be expanded) ---
        # Common wavelets suitable for signal processing
        self._available_wavelets = [
            'db2', 'db4', 'db6', 'db8', 'db10', # Daubechies
            'sym2', 'sym4', 'sym6', 'sym8', 'sym10', # Symlets
            'coif1', 'coif2', 'coif3', 'coif4', 'coif5', # Coiflets
            'bior1.3', 'bior3.7', 'bior6.8', # Biorthogonal
            'rbio3.7' # Reverse Biorthogonal
        ]

        try:
            self._init_ui()
            self._load_defaults()
            self._update_parameter_visibility(initial_setup=True) # Set initial visibility
        except Exception as e:
            logging.error(f"Error during ProcessingControlPanel initialization: {e}", exc_info=True)
            self.setEnabled(False) # Disable the widget on init error

    def _init_ui(self):
        """Initializes the UI components and layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)

        # --- Baseline Correction Parameters ---
        self.baseline_box = CollapsibleBox("1. Baseline Correction", self, is_expanded=True)
        baseline_content = QWidget()
        # Store reference to the QFormLayout ## STORED ##
        self.baseline_layout = QFormLayout(baseline_content)
        self.baseline_layout.setContentsMargins(8, 8, 8, 8)
        self.baseline_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.baseline_layout.setHorizontalSpacing(10)
        self.baseline_layout.setVerticalSpacing(8)

        # Baseline Method Combo
        self.baseline_method_combo = QComboBox()
        self.baseline_method_combo.addItems(["Polynomial", "SNIP", "None"])
        self.baseline_method_combo.setToolTip("Select the baseline correction algorithm.")
        self.baseline_method_combo.currentTextChanged.connect(self._update_parameter_visibility)
        self.baseline_layout.addRow("Method:", self.baseline_method_combo)

        # Polynomial Order (visible only if Polynomial selected)
        self.baseline_poly_order_spin = QSpinBox()
        self.baseline_poly_order_spin.setRange(0, 10)
        self.baseline_poly_order_spin.setToolTip("Order of the polynomial for baseline fitting.")
        # Store reference to the widget added to the row ## STORED ##
        self.poly_order_row_widget = QWidget()
        poly_order_row_layout = QHBoxLayout(self.poly_order_row_widget)
        poly_order_row_layout.setContentsMargins(0,0,0,0)
        poly_order_row_layout.addWidget(self.baseline_poly_order_spin)
        poly_order_row_layout.addWidget(InfoButton(lambda: QMessageBox.information(self, "Polynomial Order", "Sets the degree of the polynomial function used to fit the baseline points (selected by percentile). Low orders (1-3) are common.")))
        # Add the container widget to the form layout row
        self.baseline_layout.addRow("Polynomial Order:", self.poly_order_row_widget)

        # SNIP Iterations (visible only if SNIP selected)
        self.baseline_snip_iter_spin = QSpinBox()
        self.baseline_snip_iter_spin.setRange(1, 500)
        self.baseline_snip_iter_spin.setToolTip("Number of iterations (clipping window sizes) for the SNIP algorithm.")
        # Store reference to the widget added to the row ## STORED ##
        self.snip_iter_row_widget = QWidget()
        snip_iter_row_layout = QHBoxLayout(self.snip_iter_row_widget)
        snip_iter_row_layout.setContentsMargins(0,0,0,0)
        snip_iter_row_layout.addWidget(self.baseline_snip_iter_spin)
        snip_iter_row_layout.addWidget(InfoButton(lambda: QMessageBox.information(self, "SNIP Iterations", "Controls the smoothness of the SNIP baseline. Higher iterations remove broader features.")))
        # Add the container widget to the form layout row
        self.baseline_layout.addRow("SNIP Iterations:", self.snip_iter_row_widget)

        self.baseline_box.setContentLayout(self.baseline_layout) # Use the stored layout
        main_layout.addWidget(self.baseline_box)

        # --- Denoising Parameters --- ## NEW SECTION ##
        self.denoising_box = CollapsibleBox("2. Denoising", self, is_expanded=True)
        denoising_content = QWidget()
        # Store reference ## STORED ##
        self.denoising_layout = QFormLayout(denoising_content)
        self.denoising_layout.setContentsMargins(8, 8, 8, 8)
        self.denoising_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.denoising_layout.setHorizontalSpacing(10)
        self.denoising_layout.setVerticalSpacing(8)

        # Denoising Method Combo
        self.denoising_method_combo = QComboBox()
        self.denoising_method_combo.addItems(["Wavelet", "None"])
        self.denoising_method_combo.setToolTip("Select the denoising algorithm to apply after baseline correction.")
        self.denoising_method_combo.currentTextChanged.connect(self._update_parameter_visibility)
        self.denoising_layout.addRow("Method:", self.denoising_method_combo)

        # -- Wavelet Parameters (Container) --
        # Store reference ## STORED ##
        self.wavelet_params_widget = QWidget() # Container for wavelet params
        wavelet_params_layout = QFormLayout(self.wavelet_params_widget)
        wavelet_params_layout.setContentsMargins(0, 0, 0, 0) # No extra margins needed inside
        wavelet_params_layout.setSpacing(8)

        self.wavelet_type_combo = QComboBox()
        self.wavelet_type_combo.addItems(self._available_wavelets)
        self.wavelet_type_combo.setToolTip("Type of wavelet basis function (e.g., Daubechies 'db', Symlets 'sym').")
        wavelet_params_layout.addRow("Wavelet Type:", self.wavelet_type_combo)

        self.wavelet_level_spin = QSpinBox()
        self.wavelet_level_spin.setRange(1, 10) # Max level depends on data length, set dynamically?
        self.wavelet_level_spin.setToolTip("Decomposition level. Higher levels affect broader features. Auto-adjusts if needed.")
        wavelet_params_layout.addRow("Level:", self.wavelet_level_spin)

        self.wavelet_mode_combo = QComboBox()
        self.wavelet_mode_combo.addItems(["soft", "hard"])
        self.wavelet_mode_combo.setToolTip("Thresholding mode ('soft' shrinks coeffs, 'hard' sets to zero).")
        wavelet_params_layout.addRow("Mode:", self.wavelet_mode_combo)

        self.wavelet_threshold_factor_dspin = QDoubleSpinBox()
        self.wavelet_threshold_factor_dspin.setRange(0.1, 10.0)
        self.wavelet_threshold_factor_dspin.setDecimals(2)
        self.wavelet_threshold_factor_dspin.setSingleStep(0.1)
        self.wavelet_threshold_factor_dspin.setToolTip("Threshold = Factor × Estimated Noise (MAD). Higher values remove more signal.")
        wavelet_params_layout.addRow("Threshold Factor:", self.wavelet_threshold_factor_dspin)

        # Add wavelet container widget to the main denoising layout
        self.denoising_layout.addRow(self.wavelet_params_widget)
        self.denoising_box.setContentLayout(self.denoising_layout) # Use stored layout
        main_layout.addWidget(self.denoising_box)
        # -- End Denoising Section --

        # --- Smoothing Parameters ---
        self.smoothing_box = CollapsibleBox("3. Smoothing", self, is_expanded=True)
        smoothing_content = QWidget()
        # Store reference ## STORED ##
        self.smoothing_layout = QFormLayout(smoothing_content)
        self.smoothing_layout.setContentsMargins(8, 8, 8, 8)
        self.smoothing_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.smoothing_layout.setHorizontalSpacing(10)
        self.smoothing_layout.setVerticalSpacing(8)

        # Smoothing Method Combo
        self.smoothing_method_combo = QComboBox()
        self.smoothing_method_combo.addItems(["SavitzkyGolay", "None"])
        self.smoothing_method_combo.setToolTip("Select the smoothing algorithm to apply after denoising.")
        self.smoothing_method_combo.currentTextChanged.connect(self._update_parameter_visibility)
        self.smoothing_layout.addRow("Method:", self.smoothing_method_combo)

        # Savitzky-Golay Parameters (Container)
        # Store reference ## STORED ##
        self.savitzky_params_widget = QWidget() # Container
        savitzky_params_layout = QFormLayout(self.savitzky_params_widget)
        savitzky_params_layout.setContentsMargins(0, 0, 0, 0)
        savitzky_params_layout.setSpacing(8)

        self.sg_window_spin = QSpinBox()
        self.sg_window_spin.setRange(3, 101); self.sg_window_spin.setSingleStep(2) # Ensure odd steps
        self.sg_window_spin.setToolTip("Window length for Savitzky-Golay filter (odd number > polyorder).")
        savitzky_params_layout.addRow("Window Length:", self.sg_window_spin)

        self.sg_polyorder_spin = QSpinBox()
        self.sg_polyorder_spin.setRange(1, 10) # Order must be less than window
        self.sg_polyorder_spin.setToolTip("Polynomial order for Savitzky-Golay filter (< window length).")
        savitzky_params_layout.addRow("Polyorder:", self.sg_polyorder_spin)

        # Add SG container to main smoothing layout
        self.smoothing_layout.addRow(self.savitzky_params_widget)
        self.smoothing_box.setContentLayout(self.smoothing_layout) # Use stored layout
        main_layout.addWidget(self.smoothing_box)

        # --- Apply Button ---
        self.process_button = QPushButton("Apply Processing Steps")
        self.process_button.setIcon(QIcon.fromTheme("system-run", QIcon.fromTheme("process-working"))) # Example icon
        self.process_button.setToolTip("Apply the selected baseline, denoising, and smoothing steps.")
        self.process_button.clicked.connect(self._emit_process_signal)
        main_layout.addWidget(self.process_button)

        main_layout.addStretch() # Push content to the top
        self.setLayout(main_layout)

    def _load_defaults(self):
        """Loads default values from config into UI widgets."""
        logging.debug("Loading processing panel defaults.")
        try:
            # Baseline Defaults
            baseline_config = self.processing_config.get('baseline', {})
            self.baseline_method_combo.setCurrentText(baseline_config.get('default_method', 'Polynomial'))
            self.baseline_poly_order_spin.setValue(baseline_config.get('poly_order', 3))
            self.baseline_snip_iter_spin.setValue(baseline_config.get('snip_iterations', 100))

            # Denoising Defaults ## NEW SECTION ##
            denoising_config = self.processing_config.get('denoising', {})
            self.denoising_method_combo.setCurrentText(denoising_config.get('default_method', 'Wavelet'))
            wavelet_config = denoising_config.get('wavelet', {})
            default_wavelet_type = wavelet_config.get('wavelet_type', 'db8')
            if default_wavelet_type in self._available_wavelets:
                 self.wavelet_type_combo.setCurrentText(default_wavelet_type)
            else:
                 logging.warning(f"Default wavelet type '{default_wavelet_type}' from config not in available list. Using first available.")
                 if self._available_wavelets:
                      self.wavelet_type_combo.setCurrentIndex(0)
            self.wavelet_level_spin.setValue(wavelet_config.get('level', 4))
            self.wavelet_mode_combo.setCurrentText(wavelet_config.get('mode', 'soft'))
            self.wavelet_threshold_factor_dspin.setValue(wavelet_config.get('threshold_sigma_factor', 3.0))

            # Smoothing Defaults
            smoothing_config = self.processing_config.get('smoothing', {})
            self.smoothing_method_combo.setCurrentText(smoothing_config.get('default_method', 'SavitzkyGolay'))
            sg_config = smoothing_config.get('savitzky_golay', {})
            self.sg_window_spin.setValue(sg_config.get('window_length', 11))
            self.sg_polyorder_spin.setValue(sg_config.get('polyorder', 3))

            # Set initial visibility based on defaults
            self._update_parameter_visibility(initial_setup=True)

        except Exception as e:
            logging.error(f"Error loading processing panel defaults: {e}", exc_info=True)


    def _on_hover(self, event):
        """ Handles mouse motion events over the canvas for hover effects. """
        if not event.inaxes == self.ax:
            if self.annot.get_visible(): self.annot.set_visible(False); self._redraw_canvas()
            return

        target_element: Optional[Artist] = None; info: Dict[str, Any] = {}; min_dist_sq = float('inf')
        peak_scatters = [s for s in [self._plot_elements.get('det'), self._plot_elements.get('fit')] if isinstance(s, PathCollection)]
        # --- Check Scatter Points First ---
        for scatter in peak_scatters:
            contains, ind_dict = scatter.contains(event)
            if contains:
                # ... (rest of scatter point handling remains the same) ...
                scatter_indices = ind_dict['ind']
                if len(scatter_indices) > 0:
                    offsets = scatter.get_offsets(); points_in_radius = offsets[scatter_indices]
                    distances = np.sum((points_in_radius - [event.xdata, event.ydata])**2, axis=1); closest_idx_in_subset = np.argmin(distances)
                    scatter_index = scatter_indices[closest_idx_in_subset]
                    if scatter_index < len(offsets):
                        target_element = scatter; pos = offsets[scatter_index]; info = {'x': pos[0], 'y': pos[1]}
                        peak_list_indices = getattr(scatter, 'peak_list_indices', None)
                        if peak_list_indices and isinstance(peak_list_indices, list) and scatter_index < len(peak_list_indices):
                            original_list_index = peak_list_indices[scatter_index]
                            if 0 <= original_list_index < len(self._peaks_ref):
                                peak_obj = self._peaks_ref[original_list_index]
                                if peak_obj and hasattr(peak_obj, 'to_dataframe_row'): info['peak_info'] = peak_obj.to_dataframe_row()
                                else: info['peak_info'] = {'Index': original_list_index}
                        break # Found scatter point

        # --- Check Lines ONLY if no scatter point was found ---
        if target_element is None:
            lines_to_check: List[Line2D] = []
            for key in ['proc', 'raw', 'base']:
                 line = self._plot_elements.get(key);
                 if isinstance(line, Line2D): lines_to_check.append(line)
            fits_dict = self._plot_elements.get('fits')
            if isinstance(fits_dict, dict): lines_to_check.extend(l for l in fits_dict.values() if isinstance(l, Line2D))

            for line in lines_to_check:
                 # --- FIX: Remove radius argument ---
                 contains, ind_dict = line.contains(event) # No radius here
                 # --- End Fix ---
                 if contains:
                     # Determine nearest point manually (more reliable than line.contains with radius)
                     x_data, y_data = line.get_data()
                     if len(x_data) == 0: continue # Skip empty lines

                     # Transform event coords to data coords if needed (event.xdata/ydata are usually already data)
                     # Calculate squared distance in data coordinates
                     distances_sq = (x_data - event.xdata)**2 + (y_data - event.ydata)**2

                     # Check distance against a threshold (in data units squared) - This replaces the radius check
                     # Define a suitable threshold (e.g., equivalent to 5 display pixels)
                     # This is approximate and might need tuning
                     pixel_threshold = 5
                     x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
                     y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
                     ax_width_pixels, ax_height_pixels = self.ax.get_window_extent().size
                     try:
                         x_threshold = (x_range / ax_width_pixels * pixel_threshold) if ax_width_pixels > 0 else 0.01 * x_range
                         y_threshold = (y_range / ax_height_pixels * pixel_threshold) if ax_height_pixels > 0 else 0.01 * y_range
                         dist_threshold_sq = x_threshold**2 + y_threshold**2 # Simple bounding box approx
                     except Exception: # Handle division by zero etc.
                         dist_threshold_sq = (0.01 * x_range)**2 + (0.01 * y_range)**2 # Fallback


                     valid_indices = np.where(distances_sq <= dist_threshold_sq)[0]

                     if len(valid_indices) > 0:
                          # Find the absolutely closest index among those within threshold
                          closest_idx_in_threshold = valid_indices[np.argmin(distances_sq[valid_indices])]
                          current_dist_sq = distances_sq[closest_idx_in_threshold]

                          # If this line point is closer than any previous candidate
                          if current_dist_sq < min_dist_sq:
                               min_dist_sq = current_dist_sq
                               target_element = line
                               info = {'x': x_data[closest_idx_in_threshold], 'y': y_data[closest_idx_in_threshold]}


        # --- Update Annotation ---
        if target_element:
            self._update_annotation(target_element, info)
            self._redraw_canvas()
        elif self.annot.get_visible():
            # Hide annotation if mouse is not over any relevant element
            self.annot.set_visible(False)
            self._redraw_canvas()


    def get_settings(self) -> dict:
        """Collects current processing settings from the UI widgets."""
        settings = {}
        try:
            # Baseline
            baseline_method = self.baseline_method_combo.currentText()
            settings['baseline_method'] = baseline_method
            if baseline_method == "Polynomial":
                settings['poly_order'] = self.baseline_poly_order_spin.value()
                settings['percentile'] = self.processing_config.get('baseline', {}).get('percentile', 10.0) # Get percentile from config
            elif baseline_method == "SNIP":
                settings['max_iterations'] = self.baseline_snip_iter_spin.value() # Use correct key for SNIP iter
                settings['increasing_window'] = self.processing_config.get('baseline', {}).get('snip_increasing_window', True) # Get from config

            # Denoising ## NEW SECTION ##
            denoising_method = self.denoising_method_combo.currentText()
            settings['denoising_method'] = denoising_method
            if denoising_method == "Wavelet":
                settings['wavelet_type'] = self.wavelet_type_combo.currentText()
                settings['level'] = self.wavelet_level_spin.value()
                settings['mode'] = self.wavelet_mode_combo.currentText()
                settings['threshold_sigma_factor'] = self.wavelet_threshold_factor_dspin.value()

            # Smoothing
            smoothing_method = self.smoothing_method_combo.currentText()
            settings['smoothing_method'] = smoothing_method
            if smoothing_method == "SavitzkyGolay":
                settings['smoothing_window'] = self.sg_window_spin.value() # Renamed key for clarity
                settings['smoothing_polyorder'] = self.sg_polyorder_spin.value() # Renamed key for clarity
                # Basic validation check
                if settings['smoothing_window'] <= settings['smoothing_polyorder']:
                     logging.warning(f"Savitzky-Golay window length ({settings['smoothing_window']}) should be greater than polyorder ({settings['smoothing_polyorder']}).")
                if settings['smoothing_window'] % 2 == 0:
                     logging.warning(f"Savitzky-Golay window length ({settings['smoothing_window']}) should be odd.")

            logging.debug(f"Gathered processing settings: {settings}")
            return settings

        except Exception as e:
            logging.error(f"Error gathering processing settings: {e}", exc_info=True)
            QMessageBox.critical(self, "Settings Error", f"Could not retrieve processing settings:\n{e}")
            return {} # Return empty dict on error

    @pyqtSlot()
    def _emit_process_signal(self):
        """Validates settings and emits the process_triggered signal."""
        settings = self.get_settings()
        if settings: # Only emit if settings were gathered successfully
            logging.info("Apply Processing button clicked. Emitting signal.")
            # Perform quick validation specific to SavGol here if needed
            if settings.get('smoothing_method') == 'SavitzkyGolay':
                 wl = settings.get('smoothing_window', 0) # Use updated key
                 po = settings.get('smoothing_polyorder', 0) # Use updated key
                 if wl % 2 == 0:
                      reply = QMessageBox.warning(self, "SG Window Even",
                                                  f"Savitzky-Golay Window Length ({wl}) should be odd.\n"
                                                  "The processing function will attempt to adjust it.\n\nProceed anyway?",
                                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                                  QMessageBox.StandardButton.Yes)
                      if reply == QMessageBox.StandardButton.No: return
                 if wl <= po:
                      QMessageBox.critical(self, "SG Settings Invalid",
                                           f"Savitzky-Golay Window Length ({wl}) must be greater than Polynomial Order ({po}).\n"
                                           "Please correct the settings.")
                      return # Stop emission if settings are invalid

            self.process_triggered.emit(settings)
        else:
            logging.error("Processing signal not emitted due to settings error.")


    def setEnabled(self, enabled: bool):
        """Overrides setEnabled to also control the main action button."""
        super().setEnabled(enabled)
        try:
            # Ensure the process button reflects the overall enabled state
            self.process_button.setEnabled(enabled)
        except Exception as e:
             logging.error(f"Error in setEnabled override: {e}", exc_info=True)


# --- END OF MODIFIED FILE libs_cosmic_forge/ui/views/control_panel_view.py ---