"""
FMCW Radar Simulator — 진입점

실행: python -m app.main  (FMCW 디렉토리에서)
또는: python app/main.py
"""
import sys
import os
from pathlib import Path

# ── matplotlib 백엔드: PyQt5 임포트 전에 설정 ──
import matplotlib
matplotlib.use('QtAgg')

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontDatabase, QFont
from PyQt5.QtCore import Qt

from .ui.main_window import MainWindow

# ── 스타일시트 경로 ──
_QSS_PATH = Path(__file__).parent / 'ui' / 'style' / 'dark_radar.qss'


def load_stylesheet(app: QApplication) -> None:
    if _QSS_PATH.exists():
        with open(_QSS_PATH, encoding='utf-8') as f:
            app.setStyleSheet(f.read())
    else:
        print(f'[경고] QSS 파일을 찾을 수 없음: {_QSS_PATH}')


def main():
    # High-DPI 지원
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName('FMCW Radar Simulator')

    # 폰트 설정 (Courier New 우선)
    font = QFont('Courier New', 11)
    app.setFont(font)

    load_stylesheet(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
