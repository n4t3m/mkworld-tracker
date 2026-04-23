from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from mktracker.debug_config import load_debug_mode
from mktracker.ui.main_window import MainWindow

_APP_ICON_PATH = Path(__file__).resolve().parents[2] / "assets" / "app_icon.png"
# Unique AppUserModelID so Windows groups our windows under our icon,
# not under the generic python.exe taskbar entry.
_APP_USER_MODEL_ID = "nate.mkworldtracker"


def _set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            _APP_USER_MODEL_ID,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to set Windows AppUserModelID", exc_info=True,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if load_debug_mode():
        logging.getLogger(__name__).info(
            "Debug mode is ENABLED — extra per-race frames will be saved.",
        )
    else:
        logging.getLogger(__name__).info("Debug mode is disabled.")

    _set_windows_app_user_model_id()

    app = QApplication(sys.argv)
    if _APP_ICON_PATH.exists():
        icon = QIcon(str(_APP_ICON_PATH))
        app.setWindowIcon(icon)
    else:
        logging.getLogger(__name__).warning(
            "App icon not found at %s — using Qt default", _APP_ICON_PATH,
        )

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
