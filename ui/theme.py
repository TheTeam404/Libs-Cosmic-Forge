# -*- coding: utf-8 -*-
"""
Theme Management for the LIBS Cosmic Forge UI.
Loads and applies Qt Stylesheets (QSS) from the 'assets/styles' directory
and sets corresponding Matplotlib styles based on configuration.
"""
import os
import logging
from pathlib import Path # Use pathlib for consistency
from typing import List, Optional, Dict

# --- PyQt Imports ---
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QFile, QTextStream # Keep for potential resource loading later
except ImportError as e:
    logging.critical(f"CRITICAL ERROR in theme.py: Cannot import PyQt6 components: {e}.")
    raise ImportError(f"PyQt6 components failed to import in theme: {e}") from e

# --- Matplotlib Import ---
# Try importing matplotlib safely for styling
MPL_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors # For color checking
    MPL_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib not found. Plot styling will be basic.")
    # Define dummy plt if needed elsewhere, or rely on MPL_AVAILABLE check
    class plt: # Dummy class
         class style: @staticmethod # type: ignore
         def use(*args, **kwargs): pass
         @staticmethod
         def rcParams(*args, **kwargs): return {} # Dummy rcParams

# --- Helper Import ---
try:
     from utils.helpers import get_project_root
except ImportError:
     logging.error("Failed to import get_project_root from utils.helpers. ThemeManager may fail.")
     # Define fallback if needed (e.g., assuming script location)
     def get_project_root(): return str(Path(__file__).parent.parent.parent)

class ThemeManager:
    """Handles loading and applying QSS themes and Matplotlib styles."""

    def __init__(self, application: QApplication, config: Dict):
        """
        Initializes the ThemeManager.

        Args:
            application (QApplication): The main application instance.
            config (dict): The application's configuration dictionary, expected
                           to contain an 'appearance' section.
        """
        if application is None:
            raise ValueError("ThemeManager requires a valid QApplication instance.")

        self.app = application
        self.config = config
        self.styles_dir = Path(get_project_root()) / "assets" / "styles"
        self.available_themes = self._find_available_themes()
        self.current_theme_name: Optional[str] = None

        # Determine initial theme name
        appearance_config = self.config.get('appearance', {})
        config_default_theme = appearance_config.get('default_theme', 'dark_cosmic') # Sensible fallback

        if not self.available_themes:
            logging.error(f"No theme files (.qss) found in styles directory: '{self.styles_dir}'. UI will use default Qt style.")
            self.current_theme_name = None # No theme can be applied
        elif config_default_theme in self.available_themes:
            self.current_theme_name = config_default_theme
        else:
            # If default from config doesn't exist, use the first available theme found
            fallback_theme = self.available_themes[0]
            logging.warning(f"Default theme '{config_default_theme}' not found in available themes {self.available_themes}. Using '{fallback_theme}' instead.")
            self.current_theme_name = fallback_theme

        logging.info(f"ThemeManager Initialized. Available themes: {self.available_themes}. Current/Default: {self.current_theme_name or 'None'}")


    def _find_available_themes(self) -> List[str]:
        """Scans the styles directory for available .qss theme files."""
        themes: List[str] = []
        if not self.styles_dir.is_dir():
            logging.warning(f"Styles directory not found or not a directory: '{self.styles_dir}'")
            return themes
        try:
            for item in self.styles_dir.iterdir():
                 # Check if it's a file ending with .qss
                 if item.is_file() and item.suffix.lower() == ".qss":
                      themes.append(item.stem) # Add theme name without extension
            return sorted(themes)
        except OSError as e:
            logging.error(f"Error scanning styles directory '{self.styles_dir}': {e}", exc_info=True)
            return [] # Return empty list on error


    def get_available_themes(self) -> List[str]:
        """Returns the list of theme names found in the styles directory."""
        return self.available_themes


    def _load_stylesheet(self, theme_name: str) -> Optional[str]:
        """Loads the QSS content from the specified theme file."""
        if theme_name not in self.available_themes:
            logging.error(f"Cannot load stylesheet: Theme '{theme_name}' is not available in {self.available_themes}.")
            return None

        qss_path = self.styles_dir / f"{theme_name}.qss"
        if not qss_path.is_file():
            logging.error(f"Stylesheet file not found: '{qss_path}'")
            return None

        try:
            with open(qss_path, 'r', encoding='utf-8') as f:
                stylesheet = f.read()

            if not stylesheet:
                 logging.warning(f"Stylesheet file '{qss_path}' is empty.")
                 # Return empty string or None? Let's return None to indicate load issue.
                 return None

            # --- Simple Variable Substitution (Optional) ---
            # Example: Replace ##ICON_PATH## placeholder with actual path
            # Ensure paths use forward slashes for QSS compatibility
            try:
                 icon_path_str = (Path(get_project_root()) / "assets" / "icons").as_posix()
                 stylesheet = stylesheet.replace("##ICON_PATH##", icon_path_str)
            except Exception as e_path:
                 logging.error(f"Could not resolve icon path for stylesheet substitution: {e_path}")

            logging.debug(f"Successfully loaded stylesheet content from: {qss_path}")
            return stylesheet

        except IOError as e_io:
            logging.error(f"I/O error reading stylesheet '{qss_path}': {e_io}", exc_info=True)
            return None
        except Exception as e:
            logging.error(f"Unexpected error reading stylesheet '{qss_path}': {e}", exc_info=True)
            return None


    def apply_theme(self, theme_name: Optional[str] = None) -> bool:
        """
        Applies the specified theme (QSS and Matplotlib style).

        If theme_name is None or invalid, applies the current theme.

        Args:
            theme_name (Optional[str]): The name of the theme to apply (without .qss extension).

        Returns:
            bool: True if the theme was applied successfully, False otherwise.
        """
        target_theme = theme_name if theme_name and theme_name in self.available_themes else self.current_theme_name

        if not target_theme:
            logging.error("No valid theme available to apply (current theme is None and no valid name provided).")
            return False

        logging.info(f"Attempting to apply theme: '{target_theme}'")

        # --- Apply Qt Stylesheet ---
        stylesheet = self._load_stylesheet(target_theme)
        qss_applied = False
        if stylesheet is not None:
            try:
                self.app.setStyleSheet(stylesheet)
                # Check if stylesheet was actually applied (basic check)
                # Qt might silently ignore invalid QSS rules.
                if not self.app.styleSheet():
                    logging.warning(f"Applied stylesheet for theme '{target_theme}', but "
                                    "QApplication stylesheet is empty. QSS might contain errors.")
                self.current_theme_name = target_theme # Update current theme only on success
                qss_applied = True
                logging.info(f"Applied Qt stylesheet for theme: '{target_theme}'")
            except Exception as e_apply:
                logging.error(f"Error applying stylesheet for theme '{target_theme}': {e_apply}", exc_info=True)
        else:
            logging.error(f"Could not apply theme '{target_theme}': Stylesheet failed to load.")
            # Should we revert to default Qt style? Or keep previous? Keep previous for now.

        # --- Apply Matplotlib Style ---
        if MPL_AVAILABLE:
             try:
                 appearance_config = self.config.get('appearance', {})
                 plotting_config = appearance_config.get('plotting', {})
                 is_dark = 'dark' in target_theme.lower() # Simple check based on name

                 # Determine matplotlib style name from config
                 mpl_style_key = 'matplotlib_style_dark' if is_dark else 'matplotlib_style_light'
                 fallback_style = 'dark_background' if is_dark else 'default' # Sensible fallbacks
                 mpl_style_name = plotting_config.get(mpl_style_key, fallback_style)

                 # Apply the style
                 plt.style.use(mpl_style_name)
                 logging.info(f"Applied Matplotlib style: '{mpl_style_name}' for theme '{target_theme}'")
             except Exception as e_mpl:
                 logging.warning(f"Could not apply configured Matplotlib style '{mpl_style_name}'. Error: {e_mpl}. Using 'default'.")
                 try:
                      plt.style.use('default') # Fallback to ensure a valid style
                 except Exception as e_mpl_fallback:
                      logging.error(f"Failed to apply even fallback Matplotlib style 'default': {e_mpl_fallback}")
        else:
            logging.debug("Matplotlib not available, skipping style setting.")

        return qss_applied # Return True if at least QSS was applied (even if MPL failed)