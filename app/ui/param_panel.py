"""
레이더 파라미터 패널 (왼쪽 고정 패널)
슬라이더 변경 → params_changed 시그널 발생
"""
from __future__ import annotations
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QGroupBox, QScrollArea, QFrame, QSizePolicy
)

from ..core.models import RadarParams

# ─────────────────────────────────────────────
#  슬라이더 행 위젯
# ─────────────────────────────────────────────
class ParamRow(QWidget):
    """
    [label]  [────●────]  [value unit]
    """
    value_changed = pyqtSignal(float)

    def __init__(self, label: str, unit: str,
                 lo: float, hi: float, steps: int,
                 init: float, fmt: str = '.1f',
                 log_scale: bool = False):
        super().__init__()
        self._lo, self._hi = lo, hi
        self._steps = steps
        self._fmt = fmt
        self._log = log_scale

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(1)

        # 상단: 이름 + 값
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._lbl_name = QLabel(label)
        self._lbl_name.setStyleSheet("color:#007a20; font-size:11px;")
        self._lbl_val = QLabel(self._fmt_val(init))
        self._lbl_val.setObjectName("value")
        self._lbl_val.setAlignment(Qt.AlignRight)
        self._lbl_unit = QLabel(unit)
        self._lbl_unit.setObjectName("unit")
        self._lbl_unit.setFixedWidth(32)
        top.addWidget(self._lbl_name)
        top.addStretch()
        top.addWidget(self._lbl_val)
        top.addWidget(self._lbl_unit)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, steps)
        self._slider.setValue(self._to_tick(init))
        self._slider.setFixedHeight(16)

        layout.addLayout(top)
        layout.addWidget(self._slider)

        self._slider.valueChanged.connect(self._on_change)

    def _to_tick(self, val: float) -> int:
        if self._log:
            val = np.log10(max(val, 1e-30))
            lo = np.log10(self._lo)
            hi = np.log10(self._hi)
        else:
            lo, hi = self._lo, self._hi
        t = int(round((val - lo) / (hi - lo) * self._steps))
        return max(0, min(self._steps, t))

    def _from_tick(self, tick: int) -> float:
        frac = tick / self._steps
        if self._log:
            lo = np.log10(self._lo)
            hi = np.log10(self._hi)
            return 10 ** (lo + frac * (hi - lo))
        return self._lo + frac * (self._hi - self._lo)

    def _fmt_val(self, v: float) -> str:
        return format(v, self._fmt)

    def _on_change(self, tick: int):
        v = self._from_tick(tick)
        self._lbl_val.setText(self._fmt_val(v))
        self.value_changed.emit(v)

    def get_value(self) -> float:
        return self._from_tick(self._slider.value())

    def set_value(self, v: float, block: bool = False):
        if block:
            self._slider.blockSignals(True)
        self._slider.setValue(self._to_tick(v))
        self._lbl_val.setText(self._fmt_val(v))
        if block:
            self._slider.blockSignals(False)


# ─────────────────────────────────────────────
#  파라미터 패널
# ─────────────────────────────────────────────
class ParamPanel(QWidget):
    """
    슬라이더 기반 레이더 파라미터 편집 패널
    200ms 디바운스 후 params_changed 시그널 발생
    """
    params_changed = pyqtSignal(object)   # RadarParams

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(420)

        # 디바운스 타이머
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(180)
        self._timer.timeout.connect(self._emit)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # 스크롤 가능 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        inner_w = QWidget()
        self._vbox = QVBoxLayout(inner_w)
        self._vbox.setContentsMargins(4, 4, 4, 4)
        self._vbox.setSpacing(6)

        scroll.setWidget(inner_w)
        outer.addWidget(scroll)

        self._rows: dict[str, ParamRow] = {}
        self._build()
        self._vbox.addStretch()

    def _build(self):
        # ── 파형 ──
        grp = self._group("[ WAVEFORM ]")
        self._add(grp, 'fc_ghz',   'fc',   'GHz', 60, 81,   500, 77.0,     '.1f')
        self._add(grp, 'bw_mhz',   'B',    'MHz', 50, 500,  450, 150.0,    '.0f')
        self._add(grp, 'tc_us',    'Tc',   'μs',  5,  200,  390, 40.0,     '.0f')
        self._add(grp, 'prf_hz',   'PRF',  'Hz',  200,5000, 480, 1000.0,   '.0f')

        # ── 샘플링 ──
        grp = self._group("[ SAMPLING ]")
        self._add(grp, 'fs_mhz',   'fs',   'MHz', 5, 50,    450, 15.0,     '.1f')
        self._add(grp, 'n_chirp',  'N_c',  '',    16, 256,  240, 64.0,     '.0f')

        # ── MIMO ──
        grp = self._group("[ MIMO ]")
        self._add(grp, 'n_tx',     'N_TX', '',    1,  4,    3,   2.0,      '.0f')
        self._add(grp, 'n_rx',     'N_RX', '',    1,  8,    7,   4.0,      '.0f')

        # ── CFAR ──
        grp = self._group("[ CFAR ]")
        self._add(grp, 'pfa',      'Pfa',  '',    1e-6, 1e-2, 500, 1e-4,   '.1e',
                  log_scale=True)
        self._add(grp, 'guard',    'Guard','cell',1,  8,    7,   2.0,      '.0f')
        self._add(grp, 'train',    'Train','cell',4,  32,   28,  8.0,      '.0f')

        # ── 잡음 ──
        grp = self._group("[ NOISE ]")
        self._add(grp, 'noise_db', 'N₀',  'dB',  -80, -20, 600, -74.0,   '.0f')

    def _group(self, title: str) -> QVBoxLayout:
        gb = QGroupBox(title)
        gb.setStyleSheet(
            "QGroupBox { font-size:11px; color:#00ff41; "
            "border:1px solid #0d2a0d; border-radius:3px; "
            "margin-top:10px; padding-top:6px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:6px; "
            "padding:0 3px; color:#00ff41; }"
        )
        vb = QVBoxLayout(gb)
        vb.setContentsMargins(6, 4, 6, 4)
        vb.setSpacing(4)
        self._vbox.addWidget(gb)
        return vb

    def _add(self, parent_layout, key, label, unit,
             lo, hi, steps, init, fmt, log_scale=False):
        row = ParamRow(label, unit, lo, hi, steps, init, fmt,
                       log_scale=log_scale)
        row.value_changed.connect(self._on_any_change)
        self._rows[key] = row
        parent_layout.addWidget(row)

    def _on_any_change(self, _):
        self._timer.start()

    def _emit(self):
        self.params_changed.emit(self.get_params())

    # ── 공개 API ──
    def get_params(self) -> RadarParams:
        p = RadarParams()
        p.fc_hz          = self._rows['fc_ghz'].get_value() * 1e9
        p.bandwidth_hz   = self._rows['bw_mhz'].get_value() * 1e6
        p.chirp_dur_s    = self._rows['tc_us'].get_value() * 1e-6
        p.prf_hz         = self._rows['prf_hz'].get_value()
        p.fs_hz          = self._rows['fs_mhz'].get_value() * 1e6
        p.n_chirp        = max(16, int(self._rows['n_chirp'].get_value()))
        p.n_tx           = max(1, int(self._rows['n_tx'].get_value()))
        p.n_rx           = max(1, int(self._rows['n_rx'].get_value()))
        p.cfar_pfa       = self._rows['pfa'].get_value()
        p.cfar_guard     = max(1, int(self._rows['guard'].get_value()))
        p.cfar_train     = max(4, int(self._rows['train'].get_value()))
        noise_db         = self._rows['noise_db'].get_value()
        p.noise_std      = 10 ** (noise_db / 20.0)
        return p

    def set_params(self, p: RadarParams):
        """외부에서 파라미터 로드 시 슬라이더 동기화"""
        self._rows['fc_ghz'].set_value(p.fc_hz / 1e9, block=True)
        self._rows['bw_mhz'].set_value(p.bandwidth_hz / 1e6, block=True)
        self._rows['tc_us'].set_value(p.chirp_dur_s * 1e6, block=True)
        self._rows['prf_hz'].set_value(p.prf_hz, block=True)
        self._rows['fs_mhz'].set_value(p.fs_hz / 1e6, block=True)
        self._rows['n_chirp'].set_value(float(p.n_chirp), block=True)
        self._rows['n_tx'].set_value(float(p.n_tx), block=True)
        self._rows['n_rx'].set_value(float(p.n_rx), block=True)
        self._rows['pfa'].set_value(p.cfar_pfa, block=True)
        self._rows['guard'].set_value(float(p.cfar_guard), block=True)
        self._rows['train'].set_value(float(p.cfar_train), block=True)
        noise_db = 20 * np.log10(max(p.noise_std, 1e-12))
        self._rows['noise_db'].set_value(noise_db, block=True)
