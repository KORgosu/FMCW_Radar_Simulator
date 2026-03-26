"""
메인 윈도우
레이아웃: [파라미터 패널] | [씬 위젯] | [탭 영역]
"""
from __future__ import annotations
from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTabWidget, QLabel, QFrame, QSizePolicy
)
from PyQt5.QtGui import QFont, QIcon

from .param_panel import ParamPanel
from .scene_widget import SceneWidget, TargetInfoPanel
from .tabs import Tab1Widget, Tab2Widget, Tab3Widget, Tab4Widget

from ..core.models import RadarParams, Target
from ..core.simulator import FMCWSimulator, SimResult


class StatusBar(QWidget):
    """하단 상태 바"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(12)
        self._lbl = QLabel('READY')
        self._lbl.setStyleSheet("color:#005a15; font-size:9px;")
        self._lbl_params = QLabel('')
        self._lbl_params.setStyleSheet("color:#004a10; font-size:8px;")
        layout.addWidget(QLabel('●'))
        layout.addWidget(self._lbl)
        layout.addStretch()
        layout.addWidget(self._lbl_params)

    def set_status(self, msg: str, params: RadarParams = None):
        self._lbl.setText(msg)
        if params:
            self._lbl_params.setText(params.summary().replace('\n', '  │  '))


class TitleBar(QWidget):
    """상단 타이틀 바"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("background-color: #060d06; border-bottom: 1px solid #0d2a0d;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)

        title = QLabel('FMCW RADAR SIMULATOR')
        title.setObjectName('title')
        title.setStyleSheet(
            "color:#00ff41; font-family:'Courier New'; font-size:14px; "
            "font-weight:bold; letter-spacing:4px; background:transparent;"
        )
        subtitle = QLabel('77GHz mmWave · Signal Processing Pipeline')
        subtitle.setObjectName('subtitle')
        subtitle.setStyleSheet(
            "color:#004a10; font-family:'Courier New'; font-size:9px; "
            "letter-spacing:2px; background:transparent;"
        )

        layout.addWidget(title)
        layout.addSpacing(16)
        layout.addWidget(subtitle)
        layout.addStretch()

        # 세종대 + 제작자 레이블
        info_layout = QVBoxLayout()
        info_layout.setSpacing(1)
        info_layout.setContentsMargins(0, 0, 0, 0)

        lab_univ = QLabel('세종대학교 국방레이더기술연구실')
        lab_univ.setStyleSheet(
            "color:#00aa30; font-family:'Courier New'; font-size:10px; "
            "background:transparent;"
        )
        lab_univ.setAlignment(Qt.AlignRight)

        lab_author = QLabel('made by KORgosu (Su Hwan Park)')
        lab_author.setStyleSheet(
            "color:#007a20; font-family:'Courier New'; font-size:9px; "
            "background:transparent;"
        )
        lab_author.setAlignment(Qt.AlignRight)

        info_layout.addWidget(lab_univ)
        info_layout.addWidget(lab_author)
        layout.addLayout(info_layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FMCW Radar Simulator')
        self.resize(1440, 860)
        self.setMinimumSize(1100, 700)

        self._sim = FMCWSimulator()
        self._params = RadarParams()
        self._targets: List[Target] = []
        self._last_result: SimResult = None

        # 디바운스 타이머
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(50)   # 즉시 (params_panel 이미 디바운스)
        self._update_timer.timeout.connect(self._run_update)

        self._build_ui()
        self._connect_signals()

        # 초기 타겟으로 시작
        self._scene.add_default_targets()

    # ──────────────────────────────────────────
    #  UI 구성
    # ──────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 타이틀 바
        root.addWidget(TitleBar())

        # 메인 영역 (수평 스플리터)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        # ── 왼쪽: 파라미터 + 씬 ──
        left = QWidget()
        left.setFixedWidth(428)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(2, 4, 2, 4)
        left_layout.setSpacing(4)

        self._param_panel = ParamPanel()
        left_layout.addWidget(self._param_panel)

        # 씬 위젯
        scene_label = QLabel('[ SCENE — 타겟 배치 ]')
        scene_label.setStyleSheet(
            "color:#00ff41; font-size:9px; "
            "font-family:'Courier New'; letter-spacing:1px; "
            "border-top:1px solid #0d2a0d; padding-top:4px;"
        )
        left_layout.addWidget(scene_label)

        self._scene = SceneWidget()
        self._scene.setMinimumHeight(220)
        left_layout.addWidget(self._scene)

        # 타겟 속성 편집
        self._target_info = TargetInfoPanel(self._scene)
        left_layout.addWidget(self._target_info)

        splitter.addWidget(left)

        # ── 오른쪽: 탭 위젯 ──
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._tab1 = Tab1Widget()
        self._tab2 = Tab2Widget()
        self._tab3 = Tab3Widget()
        self._tab4 = Tab4Widget()

        self._tabs.addTab(self._tab1, '[ 1 ] 신호 원리')
        self._tabs.addTab(self._tab2, '[ 2 ] Range-Doppler')
        self._tabs.addTab(self._tab3, '[ 3 ] CFAR 탐지')
        self._tabs.addTab(self._tab4, '[ 4 ] 통합 파이프라인')

        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

        # 상태 바
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#0d2a0d;")
        root.addWidget(sep)

        self._status_bar = StatusBar()
        root.addWidget(self._status_bar)

    def _connect_signals(self):
        self._param_panel.params_changed.connect(self._on_params_change)
        self._scene.targets_changed.connect(self._on_targets_change)
        self._tabs.currentChanged.connect(self._on_tab_change)

    # ──────────────────────────────────────────
    #  이벤트 핸들러
    # ──────────────────────────────────────────
    @pyqtSlot(object)
    def _on_params_change(self, params: RadarParams):
        self._params = params
        self._update_timer.start()

    @pyqtSlot(list)
    def _on_targets_change(self, targets: List[Target]):
        self._targets = targets
        self._update_timer.start()

    @pyqtSlot(int)
    def _on_tab_change(self, idx: int):
        # 탭 전환 시 최신 결과로 즉시 업데이트
        if self._last_result is not None:
            self._update_current_tab(self._last_result)
        else:
            self._update_timer.start()

    # ──────────────────────────────────────────
    #  시뮬레이션 실행
    # ──────────────────────────────────────────
    def _run_update(self):
        self._status_bar.set_status('COMPUTING...')
        try:
            result = self._sim.compute(self._targets, self._params)
            self._last_result = result
            self._update_current_tab(result)
            n_det = len(result.point_cloud)
            self._status_bar.set_status(
                f'OK  — 탐지: {n_det}개',
                self._params
            )
        except Exception as e:
            self._status_bar.set_status(f'ERROR: {e}')

    def _update_current_tab(self, result: SimResult):
        idx = self._tabs.currentIndex()
        tab = [self._tab1, self._tab2, self._tab3, self._tab4][idx]
        tab.update_plots(result, self._params, self._targets)
