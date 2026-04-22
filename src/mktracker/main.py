from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from mktracker.debug_config import load_debug_mode
from mktracker.ui.main_window import MainWindow


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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
