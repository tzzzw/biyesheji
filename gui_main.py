"""Entry point for the PySide6 desktop GUI."""

from __future__ import annotations

import sys


def main() -> int:
    """Launch the desktop GUI."""
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "PySide6 is not installed. Please install PySide6 or PySide6_Essentials first."
        ) from exc

    from gui.app import MainWindow
    from gui.config import APP_NAME

    app = QApplication(sys.argv)
    app.setApplicationDisplayName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
