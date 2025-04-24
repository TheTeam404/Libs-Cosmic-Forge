# -*- coding: utf-8 -*-
"""
Control Panel View for Spectrum Processing settings (Baseline, Denoising, Smoothing).
Uses CollapsibleBox widgets for organization.
"""
import logging
from typing import List, Optional, Dict, Any

# --- PyQt Imports ---
try:
    from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, QSpinBox,
                                 QDoubleSpinBox, QPushButton, QLabel, QHBoxLayout, QComboBox,
                                 QMessageBox, QSizePolicy) # Removed unused QCheckBox, QTable* etc.
    from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot
    from PyQt6.QtGui import QIcon
    _QT_AVAILABLE = True
except ImportError as e:
    _QT_AVAILABLE = False
    logging.critical(f"CRITICAL ERROR in control_panel_view.py: Cannot import PyQt6 components: {e}.")
    raise ImportError(f"PyQt6 components failed to import in control_panel_view: {e}") from e

# --- Local Imports ---
try:
    from ui.widgets.info_button import InfoButton
    from ui.widgets.collapsible_box import CollapsibleBox
except ImportError as e:
    logging.critical(f"CRITICAL ERROR in control_panel_view.py: Cannot import local UI components: {e}.")
    raise ImportError(f"Local UI components failed to import in control_panel_view: {e}") from e


class ProcessingControlPanel(QWidget):
    """Control panel for spectrum processing: Baseline, Denoising, Smoothing."""
    # Signal emitted to trigger processing with current settings dictionary
    process_triggered = pyqtSignal(dict)

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Get the main processing section, or an empty dict if it doesn't exist
        self.processing_config = config.get('processing', {})
        self._dark_mode: bool = False # Track theme if needed for styling

        # --- Attributes to store layout and key widget references ---
        self.baseline_layout: Optional[QFormLayout] = None
        self.denoising_layout: Optional[QFormLayout] = None
        self.smoothing_layout: Optional[QFormLayout] = None

        # Container widgets for dynamic visibility
        self.poly_order_widget_container: Optional[QWidget] = None
        self.snip_iter_widget_container: Optional[QWidget] = None
        self.wavelet_params_widget: Optional[QWidget] = None
        self.savitzky_params_widget: Optional[QWidget] = None

        # --- List of available wavelets ---
        # Common wavelets suitable for signal processing (can be expanded or read from config)
        self._available_wavelets = [
            'db2', 'db4', 'db6', 'db8', 'db10', 'db15', 'db20', # Daubechies
            'sym2', 'sym4', 'sym6', 'sym8', 'sym10','sym15', # Symlets
            'coif1', 'coif2', 'coif3', 'coif4', 'coif5', # Coiflets
            'bior1.3', 'bior2.2', 'bior3.7', 'bior6.8', # Biorthogonal
            'rbio3.7' # Reverse Biorthogonal
        ]

        if not _QT_AVAILABLE:
             logging.critical("ProcessingControlPanel cannot be initialized: PyQt6 components missing.")
             self._init_error_ui("PyQt6 Import Error")
             return

        try:
            self._init_ui()
            self._load_defaults()
            # Set initial visibility based on defaults after loading
            self._update_parameter_visibility(initial_setup=True)
        except Exception as e:
            logging.critical(f"Error during ProcessingControlPanel initialization: {e}", exc_info=True)
            self._init_error_ui(f"Initialization Error:\n{e}")

    def _init_error_ui(self, message: str):
         """Creates a simple label indicating an error if UI init fails."""
         layout = QVBoxLayout(self)
         error_label = QLabel(f"Error initializing Processing Control Panel:\n{message}\nPlease check dependencies and logs.")
         error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
         error_label.setStyleSheet("color: red; font-weight: bold;")
         layout.addWidget(error_label)
         self.setLayout(layout)
         self.setEnabled(False)

    def _init_ui(self):
        """Initializes the UI components and layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5); main_layout.setSpacing(8)

        # --- Baseline Correction Section ---
        self.baseline_box = CollapsibleBox("1. Baseline Correction", self, is_expanded=True)
        baseline_content = QWidget(); self.baseline_layout = QFormLayout(baseline_content)
        self.baseline_layout.setContentsMargins(8, 8, 8, 8); self.baseline_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); self.baseline_layout.setHorizontalSpacing(10); self.baseline_layout.setVerticalSpacing(8)
        # Method Combo
        self.baseline_method_combo = QComboBox(); self.baseline_method_combo.addItems(["Polynomial", "SNIP", "None"]); self.baseline_method_combo.setToolTip("Select baseline correction algorithm."); self.baseline_method_combo.currentTextChanged.connect(self._update_parameter_visibility)
        self.baseline_layout.addRow("Method:", self.baseline_method_combo)
        # Polynomial Order (container for hiding)
        self.poly_order_widget_container = QWidget(); poly_order_hbox = QHBoxLayout(self.poly_order_widget_container); poly_order_hbox.setContentsMargins(0,0,0,0)
        self.baseline_poly_order_spin = QSpinBox(); self.baseline_poly_order_spin.setRange(0, 10); self.baseline_poly_order_spin.setToolTip("Order of the polynomial baseline fit (0=constant, 1=linear, ...)."); poly_order_hbox.addWidget(self.baseline_poly_order_spin)
        poly_order_hbox.addWidget(InfoButton(lambda: QMessageBox.information(self, "Polynomial Order", "Sets the degree of the polynomial function used to fit the baseline points (selected by percentile). Low orders (1-3) are common.")))
        self.baseline_layout.addRow("Polynomial Order:", self.poly_order_widget_container)
        # SNIP Iterations (container for hiding)
        self.snip_iter_widget_container = QWidget(); snip_iter_hbox = QHBoxLayout(self.snip_iter_widget_container); snip_iter_hbox.setContentsMargins(0,0,0,0)
        self.baseline_snip_iter_spin = QSpinBox(); self.baseline_snip_iter_spin.setRange(1, 500); self.baseline_snip_iter_spin.setToolTip("Number of iterations for the SNIP algorithm. Higher values remove broader features."); snip_iter_hbox.addWidget(self.baseline_snip_iter_spin)
        snip_iter_hbox.addWidget(InfoButton(lambda: QMessageBox.information(self, "SNIP Iterations", "Controls the smoothness of the SNIP baseline. Higher iterations correspond to larger filter windows, removing broader features.")))
        self.baseline_layout.addRow("SNIP Iterations:", self.snip_iter_widget_container)
        self.baseline_box.setContentLayout(self.baseline_layout)
        main_layout.addWidget(self.baseline_box)

        # --- Denoising Section ---
        self.denoising_box = CollapsibleBox("2. Denoising", self, is_expanded=True)
        denoising_content = QWidget(); self.denoising_layout = QFormLayout(denoising_content)
        self.denoising_layout.setContentsMargins(8, 8, 8, 8); self.denoising_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); self.denoising_layout.setHorizontalSpacing(10); self.denoising_layout.setVerticalSpacing(8)
        # Method Combo
        self.denoising_method_combo = QComboBox(); self.denoising_method_combo.addItems(["Wavelet", "None"]); self.denoising_method_combo.setToolTip("Select denoising algorithm (applied after baseline)."); self.denoising_method_combo.currentTextChanged.connect(self._update_parameter_visibility)
        self.denoising_layout.addRow("Method:", self.denoising_method_combo)
        # Wavelet Parameters (Container)
        self.wavelet_params_widget = QWidget(); wavelet_params_form = QFormLayout(self.wavelet_params_widget); wavelet_params_form.setContentsMargins(0, 0, 0, 0); wavelet_params_form.setSpacing(8)
        self.wavelet_type_combo = QComboBox(); self.wavelet_type_combo.addItems(self._available_wavelets); self.wavelet_type_combo.setToolTip("Type of wavelet basis function."); wavelet_params_form.addRow("Wavelet Type:", self.wavelet_type_combo)
        self.wavelet_level_spin = QSpinBox(); self.wavelet_level_spin.setRange(1, 15); self.wavelet_level_spin.setToolTip("Decomposition level (auto-adjusts if > max possible)."); wavelet_params_form.addRow("Level:", self.wavelet_level_spin)
        self.wavelet_mode_combo = QComboBox(); self.wavelet_mode_combo.addItems(["soft", "hard"]); self.wavelet_mode_combo.setToolTip("Thresholding mode ('soft' = shrinkage, 'hard' = zeroing)."); wavelet_params_form.addRow("Mode:", self.wavelet_mode_combo)
        self.wavelet_threshold_factor_dspin = QDoubleSpinBox(); self.wavelet_threshold_factor_dspin.setRange(0.1, 10.0); self.wavelet_threshold_factor_dspin.setDecimals(2); self.wavelet_threshold_factor_dspin.setSingleStep(0.1); self.wavelet_threshold_factor_dspin.setToolTip("Threshold = Factor × Estimated Noise (MAD)."); wavelet_params_form.addRow("Threshold Factor:", self.wavelet_threshold_factor_dspin)
        # Add the container widget as a single row in the main denoising layout
        self.denoising_layout.addRow(self.wavelet_params_widget)
        self.denoising_box.setContentLayout(self.denoising_layout)
        main_layout.addWidget(self.denoising_box)

        # --- Smoothing Section ---
        self.smoothing_box = CollapsibleBox("3. Smoothing", self, is_expanded=True)
        smoothing_content = QWidget(); self.smoothing_layout = QFormLayout(smoothing_content)
        self.smoothing_layout.setContentsMargins(8, 8, 8, 8); self.smoothing_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); self.smoothing_layout.setHorizontalSpacing(10); self.smoothing_layout.setVerticalSpacing(8)
        # Method Combo
        self.smoothing_method_combo = QComboBox(); self.smoothing_method_combo.addItems(["SavitzkyGolay", "None"]); self.smoothing_method_combo.setToolTip("Select smoothing algorithm (applied after denoising)."); self.smoothing_method_combo.currentTextChanged.connect(self._update_parameter_visibility)
        self.smoothing_layout.addRow("Method:", self.smoothing_method_combo)
        # Savitzky-Golay Parameters (Container)
        self.savitzky_params_widget = QWidget(); savitzky_params_form = QFormLayout(self.savitzky_params_widget); savitzky_params_form.setContentsMargins(0, 0, 0, 0); savitzky_params_form.setSpacing(8)
        self.sg_window_spin = QSpinBox(); self.sg_window_spin.setRange(3, 101); self.sg_window_spin.setSingleStep(2); self.sg_window_spin.setToolTip("Window length (odd, >=3, > polyorder)."); savitzky_params_form.addRow("Window Length:", self.sg_window_spin)
        self.sg_polyorder_spin = QSpinBox(); self.sg_polyorder_spin.setRange(0, 10); self.sg_polyorder_spin.setToolTip("Polynomial order (< window length)."); savitzky_params_form.addRow("Polyorder:", self.sg_polyorder_spin)
        # Add the container widget as a single row in the main smoothing layout
        self.smoothing_layout.addRow(self.savitzky_params_widget)
        self.smoothing_box.setContentLayout(self.smoothing_layout)
        main_layout.addWidget(self.smoothing_box)

        # --- Apply Button ---
        self.process_button = QPushButton("Apply Processing Steps"); self.process_button.setIcon(QIcon.fromTheme("system-run", QIcon.fromTheme("process-working"))); self.process_button.setToolTip("Apply selected baseline, denoising, and smoothing steps in order."); self.process_button.clicked.connect(self._emit_process_signal)
        main_layout.addWidget(self.process_button)

        main_layout.addStretch(); self.setLayout(main_layout)

    def _load_defaults(self):
        """Loads default values from config into UI widgets."""
        logging.debug("Loading processing panel defaults from config.")
        try:
            # Baseline Defaults
            baseline_config = self.processing_config.get('baseline', {})
            self.baseline_method_combo.setCurrentText(baseline_config.get('default_method', 'Polynomial'))
            self.baseline_poly_order_spin.setValue(baseline_config.get('poly_order', 3))
            self.baseline_snip_iter_spin.setValue(baseline_config.get('snip_iterations', 100))
            logging.debug(f" Baseline defaults: Method={self.baseline_method_combo.currentText()}, Order={self.baseline_poly_order_spin.value()}, SNIP Iter={self.baseline_snip_iter_spin.value()}")

            # Denoising Defaults
            denoising_config = self.processing_config.get('denoising', {})
            self.denoising_method_combo.setCurrentText(denoising_config.get('default_method', 'Wavelet'))
            wavelet_config = denoising_config.get('wavelet', {})
            default_wavelet_type = wavelet_config.get('wavelet_type', 'db8')
            # Check if default wavelet from config is in our available list
            if default_wavelet_type in self._available_wavelets:
                 self.wavelet_type_combo.setCurrentText(default_wavelet_type)
            elif self._available_wavelets: # Fallback to first available if default is invalid
                 logging.warning(f"Default wavelet type '{default_wavelet_type}' from config not in available list. Using '{self._available_wavelets[0]}'.")
                 self.wavelet_type_combo.setCurrentIndex(0)
            self.wavelet_level_spin.setValue(wavelet_config.get('level', 4))
            self.wavelet_mode_combo.setCurrentText(wavelet_config.get('mode', 'soft'))
            self.wavelet_threshold_factor_dspin.setValue(wavelet_config.get('threshold_sigma_factor', 3.0))
            logging.debug(f" Denoising defaults: Method={self.denoising_method_combo.currentText()}, Wavelet={self.wavelet_type_combo.currentText()}, Lvl={self.wavelet_level_spin.value()}, Mode={self.wavelet_mode_combo.currentText()}, Factor={self.wavelet_threshold_factor_dspin.value()}")

            # Smoothing Defaults
            smoothing_config = self.processing_config.get('smoothing', {})
            self.smoothing_method_combo.setCurrentText(smoothing_config.get('default_method', 'SavitzkyGolay'))
            sg_config = smoothing_config.get('savitzky_golay', {})
            self.sg_window_spin.setValue(sg_config.get('window_length', 11))
            self.sg_polyorder_spin.setValue(sg_config.get('polyorder', 3))
            logging.debug(f" Smoothing defaults: Method={self.smoothing_method_combo.currentText()}, SG Win={self.sg_window_spin.value()}, SG Ord={self.sg_polyorder_spin.value()}")

        except Exception as e:
            logging.error(f"Error loading processing panel defaults from config: {e}", exc_info=True)
            # UI will retain default values set in _init_ui

    def _update_parameter_visibility(self, initial_setup: bool = False):
        """Shows/hides parameter widgets based on selected methods."""
        # Use current text from combos
        baseline_method = self.baseline_method_combo.currentText()
        denoising_method = self.denoising_method_combo.currentText()
        smoothing_method = self.smoothing_method_combo.currentText()

        if not initial_setup: # Avoid logging during initial setup before defaults loaded
            logging.debug(f"Updating parameter visibility: Base='{baseline_method}', Denoise='{denoising_method}', Smooth='{smoothing_method}'")

        # --- Baseline Visibility ---
        # Hide/show the *container* widget for the parameters
        if self.baseline_layout: # Check layout exists
            is_poly = (baseline_method == "Polynomial")
            is_snip = (baseline_method == "SNIP")
            if self.poly_order_widget_container: self.poly_order_widget_container.setVisible(is_poly)
            if self.snip_iter_widget_container: self.snip_iter_widget_container.setVisible(is_snip)
            # Update row visibility using QFormLayout methods (requires layout reference)
            try:
                poly_row_index = self.baseline_layout.getWidgetPosition(self.poly_order_widget_container)[0]
                self.baseline_layout.setRowVisible(poly_row_index, is_poly)
                snip_row_index = self.baseline_layout.getWidgetPosition(self.snip_iter_widget_container)[0]
                self.baseline_layout.setRowVisible(snip_row_index, is_snip)
            except Exception as e:
                # This might fail if widgets/layout aren't fully set up yet
                if not initial_setup: # Only log error after initial setup
                    logging.error(f"Error updating baseline row visibility: {e}", exc_info=False)

        # --- Denoising Visibility ---
        if self.denoising_layout and self.wavelet_params_widget:
            is_wavelet = (denoising_method == "Wavelet")
            self.wavelet_params_widget.setVisible(is_wavelet)
            try: # Update row visibility
                 wavelet_row_index = self.denoising_layout.getWidgetPosition(self.wavelet_params_widget)[0]
                 self.denoising_layout.setRowVisible(wavelet_row_index, is_wavelet)
            except Exception as e:
                 if not initial_setup: logging.error(f"Error updating denoising row visibility: {e}", exc_info=False)

        # --- Smoothing Visibility ---
        if self.smoothing_layout and self.savitzky_params_widget:
            is_sg = (smoothing_method == "SavitzkyGolay")
            self.savitzky_params_widget.setVisible(is_sg)
            try: # Update row visibility
                 sg_row_index = self.smoothing_layout.getWidgetPosition(self.savitzky_params_widget)[0]
                 self.smoothing_layout.setRowVisible(sg_row_index, is_sg)
            except Exception as e:
                 if not initial_setup: logging.error(f"Error updating smoothing row visibility: {e}", exc_info=False)


    def get_settings(self) -> Dict[str, Any]:
        """Collects current processing settings into a dictionary."""
        settings: Dict[str, Any] = {}
        try:
            # Baseline
            baseline_method = self.baseline_method_combo.currentText()
            settings['baseline_method'] = baseline_method
            if baseline_method == "Polynomial":
                # Use more specific keys
                settings['baseline_poly_order'] = self.baseline_poly_order_spin.value()
                # Get other poly params from config if they are static (like percentile)
                poly_config = self.processing_config.get('baseline', {})
                settings['baseline_poly_percentile'] = poly_config.get('percentile', 10.0)
                settings['baseline_poly_max_iter'] = poly_config.get('max_iterations', 1) # Use 1 for single pass
                settings['baseline_poly_tolerance'] = poly_config.get('tolerance', 0.001)
            elif baseline_method == "SNIP":
                settings['baseline_snip_max_iterations'] = self.baseline_snip_iter_spin.value()
                snip_config = self.processing_config.get('baseline', {})
                settings['baseline_snip_increasing_window'] = snip_config.get('increasing_window', True)

            # Denoising
            denoising_method = self.denoising_method_combo.currentText()
            settings['denoising_method'] = denoising_method
            if denoising_method == "Wavelet":
                settings['wavelet_type'] = self.wavelet_type_combo.currentText()
                settings['wavelet_level'] = self.wavelet_level_spin.value()
                settings['wavelet_mode'] = self.wavelet_mode_combo.currentText()
                settings['wavelet_threshold_sigma_factor'] = self.wavelet_threshold_factor_dspin.value()

            # Smoothing
            smoothing_method = self.smoothing_method_combo.currentText()
            settings['smoothing_method'] = smoothing_method
            if smoothing_method == "SavitzkyGolay":
                # Use more specific keys
                settings['sg_window_length'] = self.sg_window_spin.value()
                settings['sg_polyorder'] = self.sg_polyorder_spin.value()

            logging.debug(f"Gathered processing settings: {settings}")
            return settings

        except Exception as e:
            logging.error(f"Error gathering processing settings: {e}", exc_info=True)
            QMessageBox.critical(self, "Settings Error", f"Could not retrieve processing settings:\n{e}")
            return {} # Return empty dict on error

    @pyqtSlot()
    def _emit_process_signal(self):
        """Emits the process_triggered signal with current settings."""
        # Validation removed - moved to core function (smooth_savitzky_golay)
        # The MainWindow handler will catch ValueErrors from core functions.
        settings = self.get_settings()
        if settings: # Only emit if settings were gathered successfully
            logging.info("Apply Processing button clicked. Emitting signal with settings.")
            self.process_triggered.emit(settings)
        else:
            logging.error("Processing signal not emitted due to error retrieving settings.")

    def setEnabled(self, enabled: bool):
        """Overrides setEnabled to also control the main action button."""
        super().setEnabled(enabled)
        # Update the main 'Apply' button based on the overall panel state
        if hasattr(self, 'process_button'):
            self.process_button.setEnabled(enabled)