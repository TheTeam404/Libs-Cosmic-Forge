# -*- coding: utf-8 -*-
"""
View widget for CF-LIBS calculations (Electron Density via Saha-Boltzmann, Concentrations).
This view provides controls and displays results for these advanced analyses.

WARNING: The core calculation logic associated with this view (in core/cflibs.py)
         currently uses simplified approximations or placeholders. Results for Nₑ and
         Concentrations may be inaccurate. Full implementation of standard Saha-Boltzmann
         and CF-LIBS methods is required for reliable quantitative analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

# --- PyQt Imports ---
try:
    from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, QTableWidget,
                                 QHeaderView, QTableWidgetItem, QAbstractItemView, QLabel,
                                 QHBoxLayout, QPushButton, QLineEdit, QDoubleSpinBox, QMessageBox,
                                 QApplication) # Added QApplication
    from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot
    from PyQt6.QtGui import QIcon, QBrush, QColor # Added QIcon, QBrush, QColor
    _QT_AVAILABLE = True
except ImportError as e:
    _QT_AVAILABLE = False
    logging.critical(f"CRITICAL ERROR in cf_libs_view.py: Cannot import PyQt6 components: {e}.")
    # Cannot function without Qt, re-raise
    raise ImportError(f"PyQt6 components failed to import in cf_libs_view: {e}") from e

# --- Local Imports ---
try:
    from ui.widgets.info_button import InfoButton
    # Core data models/functions are not directly called but parameters passed via signals
except ImportError as e:
    logging.critical(f"CRITICAL ERROR in cf_libs_view.py: Cannot import local dependencies: {e}.")
    raise ImportError(f"Local dependencies failed to import in cf_libs_view: {e}") from e

# --- Constants ---
CONC_COL_TOOLTIPS = {
    "Element": "Element Symbol",
    "Concentration (%)": "Calculated relative concentration (normalized to 100%). Accuracy depends on CF-LIBS assumptions and implementation.",
}

class CfLibsView(QWidget):
    """Displays controls and results for CF-LIBS analysis (Nₑ, Concentrations)."""
    # Signal to request calculation of Electron Density (Ne)
    # Args: species1(str), species2(str), temperature_k(float)
    calculate_ne_requested = pyqtSignal(str, str, float)
    # Signal to request calculation of Concentrations
    # Args: temperature_k(float), ne_cm3(Optional[float]) - Use object for flexibility from signal
    calculate_conc_requested = pyqtSignal(float, object)

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config.get('cflibs', {}) # Get CF-LIBS sub-config
        # Store last known/used parameters for display and requests
        self.last_temp_k: Optional[float] = None
        self.last_ne_cm3: Optional[float] = None
        self.last_conc_df: Optional[pd.DataFrame] = None

        if not _QT_AVAILABLE:
             logging.critical("CfLibsView cannot be initialized: PyQt6 components missing.")
             self._init_error_ui("PyQt6 Import Error")
             return

        try:
            self._init_ui()
            self._connect_signals()
            self._update_button_states() # Initial state
        except Exception as e:
             logging.critical(f"Error during CfLibsView initialization: {e}", exc_info=True)
             self._init_error_ui(f"Initialization Error:\n{e}")

    def _init_error_ui(self, message: str):
         """Creates a simple label indicating an error if UI init fails."""
         layout = QVBoxLayout(self)
         error_label = QLabel(f"Error initializing CF-LIBS View:\n{message}\n\nPlease check dependencies and logs.")
         error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
         error_label.setStyleSheet("color: red; font-weight: bold;")
         layout.addWidget(error_label)
         self.setLayout(layout)
         self.setEnabled(False)

    def _init_ui(self):
        """Initializes the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10) # Slightly more spacing between groups

        # --- Electron Density (Ne) Group ---
        # Add warning about placeholder status
        ne_group = QGroupBox("Electron Density (N\u2091) - Saha Approx. [Experimental]")
        ne_layout = QFormLayout(ne_group)
        ne_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); ne_layout.setHorizontalSpacing(10); ne_layout.setVerticalSpacing(8)

        # Species Input
        self.ne_species1_input = QLineEdit(); self.ne_species1_input.setPlaceholderText("e.g., Fe I (Lower Ion State)")
        self.ne_species2_input = QLineEdit(); self.ne_species2_input.setPlaceholderText("e.g., Fe II (Higher Ion State)")
        species_info_btn = InfoButton(self._show_ne_info, tooltip_text="Help on Nₑ calculation", parent=self)
        species_hbox = QHBoxLayout(); species_hbox.addWidget(self.ne_species1_input); species_hbox.addWidget(QLabel(" & ")); species_hbox.addWidget(self.ne_species2_input); species_hbox.addWidget(species_info_btn)

        # Temperature Input (Required for Saha)
        self.ne_temp_input = QDoubleSpinBox()
        self.ne_temp_input.setRange(1000, 50000); self.ne_temp_input.setDecimals(0); self.ne_temp_input.setSuffix(" K"); self.ne_temp_input.setToolTip("Plasma temperature (Tₑ) in Kelvin, needed for Saha equation.\nObtain from Boltzmann plot or enter manually."); self.ne_temp_input.setKeyboardTracking(False)

        self.calculate_ne_button = QPushButton("Calculate N\u2091")
        self.calculate_ne_button.setToolTip("Estimate electron density using selected species and temperature.\nWARNING: Uses simplified Saha-Boltzmann approximation."); self.calculate_ne_button.setIcon(QIcon.fromTheme("view-statistics", QIcon.fromTheme("accessories-calculator")))
        self.ne_result_label = QLabel("N\u2091 Result: N/A"); self.ne_result_label.setStyleSheet("font-weight: bold; padding: 3px;")

        ne_layout.addRow("Species Pair:", species_hbox)
        ne_layout.addRow("Plasma Temp (T\u2091):", self.ne_temp_input)
        ne_layout.addRow(self.calculate_ne_button)
        ne_layout.addRow(self.ne_result_label)
        main_layout.addWidget(ne_group)

        # --- Concentration Group ---
        # Add warning about placeholder status
        conc_group = QGroupBox("Concentration (CF-LIBS) - [Experimental/Simplified]")
        conc_layout = QFormLayout(conc_group)
        conc_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); conc_layout.setHorizontalSpacing(10); conc_layout.setVerticalSpacing(8)

        # Display Inputs Used
        conc_inputs_hbox = QHBoxLayout()
        self.conc_temp_label = QLabel("Using T\u2091:"); self.conc_temp_value = QLabel("N/A"); self.conc_temp_value.setStyleSheet("font-weight: bold;")
        self.conc_ne_label = QLabel("and N\u2091:"); self.conc_ne_value = QLabel("N/A"); self.conc_ne_value.setStyleSheet("font-weight: bold;")
        conc_inputs_hbox.addWidget(self.conc_temp_label); conc_inputs_hbox.addWidget(self.conc_temp_value); conc_inputs_hbox.addSpacing(20)
        conc_inputs_hbox.addWidget(self.conc_ne_label); conc_inputs_hbox.addWidget(self.conc_ne_value); conc_inputs_hbox.addStretch()

        self.calculate_conc_button = QPushButton("Calculate Concentrations"); self.calculate_conc_button.setToolTip("Estimate relative element concentrations (CF-LIBS).\nWARNING: Uses simplified method, assumes LTE & optical thinness."); self.calculate_conc_button.setIcon(QIcon.fromTheme("view-statistics", QIcon.fromTheme("accessories-calculator")))
        conc_info_btn = InfoButton(self._show_conc_info, tooltip_text="Help on CF-LIBS calculation", parent=self)

        # Results Table
        self.conc_results_table = QTableWidget()
        self.conc_columns = ["Element", "Concentration (%)"]; self.conc_results_table.setColumnCount(len(self.conc_columns)); self.conc_results_table.setHorizontalHeaderLabels(self.conc_columns)
        self.conc_results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.conc_results_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection) # Allow copy
        self.conc_results_table.setAlternatingRowColors(True); self.conc_results_table.setSortingEnabled(True); self.conc_results_table.verticalHeader().setVisible(False)
        self.conc_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.conc_results_table.setMinimumHeight(150)

        conc_button_hbox = QHBoxLayout(); conc_button_hbox.addWidget(self.calculate_conc_button); conc_button_hbox.addStretch(); conc_button_hbox.addWidget(conc_info_btn)
        conc_layout.addRow("Input Params:", conc_inputs_hbox)
        conc_layout.addRow(conc_button_hbox)
        conc_layout.addRow(self.conc_results_table)
        main_layout.addWidget(conc_group, 1) # Give table vertical stretch

        # Removed stretch from main layout - let group boxes define size
        # main_layout.addStretch()
        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect button clicks and input changes."""
        self.calculate_ne_button.clicked.connect(self._request_ne_calculation)
        self.calculate_conc_button.clicked.connect(self._request_conc_calculation)
        # Connect valueChanged to update internal state and other labels
        self.ne_temp_input.valueChanged.connect(self._temp_input_changed)

    def _show_ne_info(self):
        QMessageBox.information(self, "Electron Density (Saha-Boltzmann)",
            "Estimates electron density (N\u2091) using the ratio of intensities "
            "from lines of two *consecutive ionization stages* of the *same element*.\n\n"
            "Requires:\n"
            "- Correct Species Pair (e.g., 'Fe I', 'Fe II').\n"
            "- Fitted lines with accurate atomic data (A\u2096\u1d62, g\u2096, E\u2096) for both stages.\n"
            "- Plasma Temperature (T\u2091) in Kelvin (typically from Boltzmann plot).\n"
            "- Ionization energy and Partition functions U(T) for both species (loaded from `database/atomic_data/` files).\n\n"
            "**WARNING:** The current implementation uses a **simplified approximation** based on averaging line properties. "
            "It does NOT perform a full Saha-Boltzmann plot analysis and may be inaccurate. "
            "Unit consistency in the formula also needs validation (Issue #19.9). Use results with caution.")

    def _show_conc_info(self):
        QMessageBox.information(self, "Concentration Estimation (CF-LIBS)",
            "Calibration-Free LIBS (CF-LIBS) estimates relative elemental "
            "composition.\n\n"
            "**Critical Assumptions:**\n"
            "- Local Thermodynamic Equilibrium (LTE).\n"
            "- Optically Thin plasma (no significant self-absorption).\n\n"
            "Requires:\n"
            "- Fitted lines with accurate atomic data (A\u2096\u1d62, g\u2096, E\u2096).\n"
            "- Plasma Temperature (T\u2091) (typically from Boltzmann plot).\n"
            "- Electron Density (N\u2091) (optional, improves accuracy if used in models).\n"
            "- Partition functions U(T) for all relevant species (from `database/atomic_data/partition_functions.csv`).\n\n"
            "**WARNING:** The current implementation uses a **simplified approximation** based on averaging line properties. "
            "It does NOT perform standard CF-LIBS summation or implement optical thinness checks. Results may be inaccurate. Use with caution.")


    def _update_button_states(self):
        """Enable/disable calculation buttons based on required inputs."""
        # Nₑ Calculation Requirements
        temp_valid_for_ne = self.last_temp_k is not None and np.isfinite(self.last_temp_k) and self.last_temp_k > 0
        sp1_str = self.ne_species1_input.text().strip()
        sp2_str = self.ne_species2_input.text().strip()
        # Basic check for "Elem Ion" format (can be improved)
        sp1_valid = len(sp1_str.split()) == 2 #and (sp1_str.split()[1].isupper() or sp1_str.split()[1].isdigit())
        sp2_valid = len(sp2_str.split()) == 2 #and (sp2_str.split()[1].isupper() or sp2_str.split()[1].isdigit())
        can_calc_ne = temp_valid_for_ne and sp1_valid and sp2_valid
        self.calculate_ne_button.setEnabled(can_calc_ne)

        # Concentration Calculation Requirements
        temp_valid_for_conc = self.last_temp_k is not None and np.isfinite(self.last_temp_k) and self.last_temp_k > 0
        # Ne is optional for this simplified implementation, so only Temp is strictly required
        can_calc_conc = temp_valid_for_conc
        self.calculate_conc_button.setEnabled(can_calc_conc)


    @pyqtSlot(float)
    def _temp_input_changed(self, value: float):
        """Update internal temperature and related UI when user changes spinbox."""
        new_temp = value if value > 0 and np.isfinite(value) else None
        # Update only if the value actually changed to avoid potential signal loops
        if self.last_temp_k is None or new_temp is None or not np.isclose(self.last_temp_k, new_temp):
             log.debug(f"Temperature input changed manually to: {new_temp}")
             self.update_temperature(new_temp) # Call update method to sync labels and button states


    def update_temperature(self, temp_k: Optional[float]):
        """Receives temperature (e.g., from Boltzmann or manual input) and updates UI."""
        valid_temp = temp_k is not None and np.isfinite(temp_k) and temp_k > 0
        self.last_temp_k = float(temp_k) if valid_temp else None # Store valid float or None
        temp_str = f"{self.last_temp_k:.0f} K" if self.last_temp_k is not None else "N/A"

        # Update spinbox without emitting signal if value differs
        current_spin_val = self.ne_temp_input.value()
        new_spin_val = self.last_temp_k if self.last_temp_k is not None else 0 # Spinbox needs numeric
        if abs(current_spin_val - new_spin_val) > 1e-3: # Tolerance for float comparison
             self.ne_temp_input.blockSignals(True)
             self.ne_temp_input.setValue(new_spin_val)
             self.ne_temp_input.blockSignals(False)

        self.conc_temp_value.setText(temp_str) # Update display label in Conc group
        self._update_button_states() # Re-check button enablement


    def update_electron_density(self, ne_cm3: Optional[float]):
        """Updates the electron density display after calculation or state restore."""
        valid_ne = ne_cm3 is not None and np.isfinite(ne_cm3) and ne_cm3 > 0
        self.last_ne_cm3 = float(ne_cm3) if valid_ne else None # Store valid float or None
        ne_str = f"{self.last_ne_cm3:.2e} cm⁻³" if self.last_ne_cm3 is not None else "N/A"

        self.ne_result_label.setText(f"N\u2091 Result: {ne_str}")
        self.conc_ne_value.setText(ne_str) # Update display label for concentration input
        self._update_button_states() # Re-check button states


    def display_concentrations(self, conc_df: Optional[pd.DataFrame]):
        """Displays the calculated concentrations in the table."""
        self.last_conc_df = conc_df
        self.conc_results_table.setSortingEnabled(False)
        self.conc_results_table.setRowCount(0) # Clear previous

        if conc_df is None or conc_df.empty:
            logging.info("No concentration results to display.")
            # Display a message in the table
            self.conc_results_table.setRowCount(1)
            item = QTableWidgetItem("(No results or calculation failed)")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable) # Read-only
            item.setForeground(QBrush(QColor('gray')))
            self.conc_results_table.setItem(0, 0, item)
            # Span the message across both columns
            self.conc_results_table.setSpan(0, 0, 1, len(self.conc_columns))
            return

        # --- Populate Table ---
        logging.info(f"Displaying {len(conc_df)} concentration results.")
        self.conc_results_table.setRowCount(len(conc_df))
        col_map = {name: i for i, name in enumerate(self.conc_columns)}

        for r_idx, row_data in conc_df.iterrows():
            # Element Column
            elem = row_data.get("Element", "?")
            item_elem = self._create_table_item(elem, alignment=Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignVCenter)
            self.conc_results_table.setItem(r_idx, col_map["Element"], item_elem)

            # Concentration Column (Format as percentage)
            conc = row_data.get("Concentration")
            item_conc = self._create_table_item(conc * 100 if conc is not None else None, precision=2) # Format as XX.XX %
            self.conc_results_table.setItem(r_idx, col_map["Concentration (%)"], item_conc)

        # --- Set Header Tooltips ---
        header = self.conc_results_table.horizontalHeader()
        for i, col_name in enumerate(self.conc_columns):
            tooltip = CONC_COL_TOOLTIPS.get(col_name, col_name)
            header_item = self.conc_results_table.horizontalHeaderItem(i)
            if header_item: header_item.setToolTip(tooltip)

        self.conc_results_table.resizeColumnsToContents()
        self.conc_results_table.setSortingEnabled(True)

    def _create_table_item(self, value: Any, precision: Optional[int] = None, scientific: bool = False, alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) -> QTableWidgetItem:
        """Helper function to create and format a QTableWidgetItem."""
        # (Same helper function as in Boltzmann view - consider moving to utils if reused more)
        item = QTableWidgetItem()
        text = ""
        data_for_sorting = None # Store numeric or string for sorting

        if value is None or (isinstance(value, (float, np.number)) and not np.isfinite(value)):
            text = "" # Display empty for invalid data
            item.setForeground(QBrush(QColor('gray')))
        elif isinstance(value, (int, np.integer)):
            text = str(value)
            data_for_sorting = int(value)
        elif isinstance(value, (float, np.floating)):
            if scientific: text = f"{value:.2e}"
            elif precision is not None: text = f"{value:.{precision}f}"
            else: text = f"{value:.4g}"
            data_for_sorting = float(value)
            item.setToolTip(str(value)) # Show full value in tooltip
        else:
            text = str(value)
            alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            data_for_sorting = text

        item.setText(text)
        item.setTextAlignment(alignment)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        if data_for_sorting is not None:
             item.setData(Qt.ItemDataRole.DisplayRole, data_for_sorting)
        return item

    @pyqtSlot()
    def _request_ne_calculation(self):
        """Gathers inputs and emits signal to calculate Ne."""
        species1 = self.ne_species1_input.text().strip()
        species2 = self.ne_species2_input.text().strip()
        temp_k = self.ne_temp_input.value() # Get current value from spinbox

        # Validate inputs before emitting
        if not species1 or len(species1.split()) != 2:
             QMessageBox.warning(self,"Input Error","Enter valid Species 1 (e.g., 'Fe I').")
             return
        if not species2 or len(species2.split()) != 2:
             QMessageBox.warning(self,"Input Error","Enter valid Species 2 (e.g., 'Fe II').")
             return
        if temp_k <= 0 or not np.isfinite(temp_k):
             QMessageBox.warning(self,"Input Error","Enter valid Plasma Temperature (Tₑ > 0 K).")
             return
        if species1.split()[0].lower() != species2.split()[0].lower():
             QMessageBox.warning(self,"Input Error","Species pair must be for the same element (e.g., Fe I and Fe II).")
             return
        # TODO: Add check that ion stages are consecutive?

        logging.info(f"Requesting Nₑ calculation: {species1}/{species2}, T={temp_k:.0f}K")
        self.parent().update_status(f"Calculating Nₑ for {species1}/{species2}...", 0) # Show busy status
        QApplication.processEvents() # Update UI
        # TODO: Consider running this in a thread (Issue 18.6)
        self.calculate_ne_requested.emit(species1, species2, temp_k)


    @pyqtSlot()
    def _request_conc_calculation(self):
        """Gathers inputs and emits signal to calculate concentrations."""
        # Use the internally stored temperature
        if self.last_temp_k is None or not np.isfinite(self.last_temp_k):
             QMessageBox.warning(self,"Input Error","Valid Plasma Temperature (Tₑ) required. Calculate using Boltzmann plot first.")
             return

        # Use internally stored Ne (which might be None)
        ne_to_pass = self.last_ne_cm3 if self.last_ne_cm3 is not None and np.isfinite(self.last_ne_cm3) else None

        logging.info(f"Requesting Concentration calculation: T={self.last_temp_k:.0f}K, Nₑ={ne_to_pass or 'N/A'}")
        self.parent().update_status("Calculating Concentrations...", 0) # Show busy status
        QApplication.processEvents() # Update UI
        # TODO: Consider running this in a thread (Issue 18.6)
        self.calculate_conc_requested.emit(self.last_temp_k, ne_to_pass)


    def clear_all(self):
        """Clears all inputs, results, and internal state in this view."""
        logging.debug("Clearing CF-LIBS view.")
        # Clear Ne inputs/results
        self.ne_species1_input.clear()
        self.ne_species2_input.clear()
        self.ne_temp_input.blockSignals(True) # Block signals during reset
        self.ne_temp_input.setValue(0)
        self.ne_temp_input.blockSignals(False)
        self.ne_result_label.setText("N\u2091 Result: N/A")
        # Clear Conc inputs/results
        self.conc_temp_value.setText("N/A")
        self.conc_ne_value.setText("N/A")
        self.conc_results_table.setRowCount(0)
        # Clear internal state
        self.last_temp_k = None
        self.last_ne_cm3 = None
        self.last_conc_df = None
        # Update button states
        self._update_button_states()

    def setEnabled(self, enabled: bool):
         """Overrides setEnabled to manage child widget states appropriately."""
         # Keep the main group boxes visible but disable/enable their contents
         # super().setEnabled(enabled) # Don't disable the whole view widget

         # Enable/disable interactive elements
         self.ne_species1_input.setEnabled(enabled)
         self.ne_species2_input.setEnabled(enabled)
         self.ne_temp_input.setEnabled(enabled)
         # Buttons depend on both overall enablement AND internal logic
         self._update_button_states() # Recalculate button state based on current inputs
         if not enabled: # Explicitly disable buttons if view is disabled externally
              self.calculate_ne_button.setEnabled(False)
              self.calculate_conc_button.setEnabled(False)

         # Clear results if the view becomes disabled
         if not enabled:
              # Only clear results, not inputs, when disabled externally
              self.ne_result_label.setText("N\u2091 Result: N/A")
              self.conc_results_table.setRowCount(0)
              # Keep last T/Ne values for display, but don't clear them fully


    def set_restored_data(self, temperature_k: Optional[float], ne_cm3: Optional[float], concentrations: Optional[pd.DataFrame]):
        """Applies state loaded from a session file."""
        logging.info("Restoring state to CF-LIBS view.")
        self.update_temperature(temperature_k)
        self.update_electron_density(ne_cm3)
        self.display_concentrations(concentrations)
        # Buttons state will be updated by the update methods