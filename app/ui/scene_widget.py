"""
타겟 배치 씬 위젯
마우스 클릭/드래그로 타겟 추가·이동·삭제
"""
from __future__ import annotations
import math
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QRadialGradient, QPainterPath, QFontMetrics
)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QComboBox, QPushButton,
    QGroupBox, QSizePolicy, QFrame
)

from ..core.models import Target

# ── RCS 프리셋 ──
RCS_PRESETS = {
    '보행자 (0.5)': 0.5,
    '자전거 (2)':   2.0,
    '승용차 (10)':  10.0,
    '트럭 (100)':   100.0,
}

# ── 색상 ──
COL_BG       = QColor('#060d06')
COL_RING     = QColor('#1a5a1a')
COL_RING_LBL = QColor('#00aa30')
COL_ANGLE    = QColor('#144414')
COL_RADAR    = QColor('#00ff41')
COL_TARGET   = QColor('#00ff41')
COL_SEL      = QColor('#66ffaa')
COL_TEXT     = QColor('#00ff41')


class SceneWidget(QWidget):
    """
    레이더 + 타겟의 top-down 뷰
    - 왼쪽 클릭 (빈 공간): 타겟 추가
    - 왼쪽 드래그: 타겟 이동
    - 우클릭: 타겟 삭제
    """
    targets_changed = pyqtSignal(list)   # List[Target]

    MAX_RANGE = 200.0  # m  (표시 최대 거리, 파라미터와 별도)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._targets: List[Target] = []
        self._selected: Optional[int] = None   # 선택된 타겟 인덱스
        self._drag_idx: Optional[int] = None   # 드래그 중인 인덱스
        self._hover_idx: Optional[int] = None

    # ─────────────────────────────────────────
    #  좌표 변환
    # ─────────────────────────────────────────
    def _radar_origin(self) -> QPointF:
        """레이더 픽셀 위치 (하단 중앙)"""
        return QPointF(self.width() / 2, self.height() - 28)

    def _scale(self) -> float:
        """m → pixel 스케일"""
        usable_h = self.height() - 50
        usable_w = self.width() - 30
        fov_w = self.MAX_RANGE * 2 * math.sin(math.radians(65))
        return min(usable_h / self.MAX_RANGE, usable_w / fov_w)

    def polar_to_px(self, r: float, angle_deg: float) -> QPointF:
        o = self._radar_origin()
        sc = self._scale()
        rad = math.radians(angle_deg)
        x = o.x() + sc * r * math.sin(rad)
        y = o.y() - sc * r * math.cos(rad)
        return QPointF(x, y)

    def px_to_polar(self, px: QPointF) -> tuple:
        o = self._radar_origin()
        sc = self._scale()
        dx = (px.x() - o.x()) / sc
        dy = (o.y() - px.y()) / sc
        r = math.sqrt(dx ** 2 + dy ** 2)
        angle_deg = math.degrees(math.atan2(dx, dy))
        return r, angle_deg

    def _hit_test(self, pos: QPointF) -> Optional[int]:
        """pos에 가장 가까운 타겟 인덱스 반환 (15px 이내)"""
        best_i, best_d = None, 20.0
        for i, tgt in enumerate(self._targets):
            tp = self.polar_to_px(tgt.range_m, tgt.angle_deg)
            d = math.sqrt((tp.x() - pos.x()) ** 2 + (tp.y() - pos.y()) ** 2)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    # ─────────────────────────────────────────
    #  마우스 이벤트
    # ─────────────────────────────────────────
    def mousePressEvent(self, e):
        pos = QPointF(e.pos())
        hit = self._hit_test(pos)

        if e.button() == Qt.LeftButton:
            if hit is not None:
                self._drag_idx = hit
                self._selected = hit
            else:
                # 새 타겟 추가
                r, ang = self.px_to_polar(pos)
                r = max(5.0, min(r, self.MAX_RANGE))
                ang = max(-65.0, min(ang, 65.0))
                self._targets.append(Target(range_m=r, angle_deg=ang))
                self._selected = len(self._targets) - 1
                self._drag_idx = self._selected
                self.targets_changed.emit(list(self._targets))
        elif e.button() == Qt.RightButton:
            if hit is not None:
                self._targets.pop(hit)
                self._selected = None
                self.targets_changed.emit(list(self._targets))

        self.update()

    def mouseMoveEvent(self, e):
        pos = QPointF(e.pos())
        self._hover_idx = self._hit_test(pos)

        if self._drag_idx is not None and e.buttons() & Qt.LeftButton:
            r, ang = self.px_to_polar(pos)
            r = max(5.0, min(r, self.MAX_RANGE))
            ang = max(-65.0, min(ang, 65.0))
            tgt = self._targets[self._drag_idx]
            tgt.range_m = r
            tgt.angle_deg = ang
            self.targets_changed.emit(list(self._targets))

        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_idx = None

    # ─────────────────────────────────────────
    #  페인트
    # ─────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        o = self._radar_origin()
        sc = self._scale()

        # ── 배경 ──
        p.fillRect(self.rect(), COL_BG)

        # ── 반원 FOV 그라디언트 ──
        grad = QRadialGradient(o.x(), o.y(), sc * self.MAX_RANGE)
        grad.setColorAt(0.0, QColor(0, 30, 0, 80))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(o.x() - sc * self.MAX_RANGE),
                      int(o.y() - sc * self.MAX_RANGE),
                      int(sc * self.MAX_RANGE * 2),
                      int(sc * self.MAX_RANGE * 2))

        # ── 거리 링 ──
        font = QFont("Courier New", 9)
        p.setFont(font)
        for r_ring in range(50, int(self.MAX_RANGE) + 1, 50):
            px_r = int(sc * r_ring)
            pen = QPen(COL_RING_LBL if r_ring % 100 == 0 else COL_RING, 0.8)
            pen.setStyle(Qt.DotLine if r_ring % 100 != 0 else Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(int(o.x()) - px_r, int(o.y()) - px_r,
                      px_r * 2, px_r * 2, 0, 180 * 16)
            # 레이블
            p.setPen(QPen(COL_RING_LBL, 0.5))
            p.drawText(int(o.x()) + 3, int(o.y()) - px_r + 4,
                       f'{r_ring}m')

        # ── 각도 선 ──
        p.setPen(QPen(COL_ANGLE, 0.6))
        for ang in range(-60, 61, 15):
            rad = math.radians(ang)
            ex = o.x() + sc * self.MAX_RANGE * math.sin(rad)
            ey = o.y() - sc * self.MAX_RANGE * math.cos(rad)
            p.drawLine(int(o.x()), int(o.y()), int(ex), int(ey))
        # 각도 레이블
        p.setPen(QPen(COL_RING_LBL, 0.5))
        for ang in [-60, -30, 0, 30, 60]:
            rad = math.radians(ang)
            lx = o.x() + (sc * self.MAX_RANGE + 8) * math.sin(rad)
            ly = o.y() - (sc * self.MAX_RANGE + 8) * math.cos(rad)
            p.drawText(int(lx) - 12, int(ly) + 4, f'{ang}°')

        # ── 타겟 ──
        for i, tgt in enumerate(self._targets):
            tp = self.polar_to_px(tgt.range_m, tgt.angle_deg)
            r_px = max(5, int(math.sqrt(tgt.rcs_m2) * 2.5))

            is_sel = (i == self._selected)
            is_hov = (i == self._hover_idx)
            col = COL_SEL if is_sel else (QColor('#33ff88') if is_hov else COL_TARGET)

            # 외곽 링
            if is_sel:
                p.setPen(QPen(col, 1.5, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(tp, r_px + 4, r_px + 4)

            # 채운 원
            p.setPen(QPen(col, 1.2))
            p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 60)))
            p.drawEllipse(tp, r_px, r_px)

            # 십자선
            p.setPen(QPen(col, 1.0))
            p.drawLine(int(tp.x()) - r_px - 3, int(tp.y()),
                       int(tp.x()) + r_px + 3, int(tp.y()))
            p.drawLine(int(tp.x()), int(tp.y()) - r_px - 3,
                       int(tp.x()), int(tp.y()) + r_px + 3)

            # 레이블
            p.setPen(QPen(COL_TEXT, 0.8))
            p.setFont(QFont("Courier New", 9))
            label = f"T{i+1} {tgt.range_m:.0f}m"
            if tgt.velocity_mps != 0:
                sign = '+' if tgt.velocity_mps > 0 else ''
                label += f" {sign}{tgt.velocity_mps:.1f}"
            p.drawText(int(tp.x()) + r_px + 4, int(tp.y()) - 2, label)

        # ── 레이더 아이콘 ──
        p.setPen(QPen(COL_RADAR, 1.5))
        p.setBrush(QBrush(COL_RADAR))
        radar_tri = QPainterPath()
        radar_tri.moveTo(o.x(), o.y() - 8)
        radar_tri.lineTo(o.x() - 7, o.y() + 4)
        radar_tri.lineTo(o.x() + 7, o.y() + 4)
        radar_tri.closeSubpath()
        p.drawPath(radar_tri)

        # ── 수평선 ──
        p.setPen(QPen(COL_RING, 0.8))
        p.drawLine(0, int(o.y()) + 10, w, int(o.y()) + 10)

        # ── 타겟 수 표시 ──
        p.setPen(QPen(QColor('#004a10'), 0.8))
        p.setFont(QFont("Courier New", 10))
        p.drawText(4, h - 4, f'TARGETS: {len(self._targets)}')

        p.end()

    # ─────────────────────────────────────────
    #  공개 API
    # ─────────────────────────────────────────
    def get_targets(self) -> List[Target]:
        return list(self._targets)

    def set_targets(self, targets: List[Target]):
        self._targets = list(targets)
        self._selected = None
        self.update()

    def get_selected(self) -> Optional[Target]:
        if self._selected is not None and self._selected < len(self._targets):
            return self._targets[self._selected]
        return None

    def update_selected(self, **kwargs):
        """선택된 타겟의 속성을 업데이트"""
        if self._selected is None:
            return
        tgt = self._targets[self._selected]
        for k, v in kwargs.items():
            setattr(tgt, k, v)
        self.targets_changed.emit(list(self._targets))
        self.update()

    def clear_targets(self):
        self._targets.clear()
        self._selected = None
        self.targets_changed.emit([])
        self.update()

    def add_default_targets(self):
        """데모용 기본 타겟 3개 추가"""
        self._targets = [
            Target(range_m=40, velocity_mps=5.0,  angle_deg=-15, rcs_m2=10.0),
            Target(range_m=80, velocity_mps=-3.0, angle_deg=20,  rcs_m2=1.0),
            Target(range_m=120, velocity_mps=0.0, angle_deg=0,   rcs_m2=50.0),
        ]
        self._selected = None
        self.targets_changed.emit(list(self._targets))
        self.update()


# ─────────────────────────────────────────────
#  타겟 속성 편집 패널 (씬 아래)
# ─────────────────────────────────────────────
class TargetInfoPanel(QWidget):
    """
    선택된 타겟의 속성을 편집하는 소형 패널
    """
    target_updated = pyqtSignal()

    def __init__(self, scene: SceneWidget, parent=None):
        super().__init__(parent)
        self._scene = scene
        self._building = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # 속도
        layout.addWidget(self._lbl('V'))
        self._spn_vel = QDoubleSpinBox()
        self._spn_vel.setRange(-50, 50)
        self._spn_vel.setSingleStep(0.5)
        self._spn_vel.setDecimals(1)
        self._spn_vel.setSuffix(' m/s')
        self._spn_vel.setFixedWidth(90)
        layout.addWidget(self._spn_vel)

        layout.addWidget(self._lbl('RCS'))
        self._cmb_rcs = QComboBox()
        self._cmb_rcs.setFixedWidth(130)
        for k in RCS_PRESETS:
            self._cmb_rcs.addItem(k)
        layout.addWidget(self._cmb_rcs)

        # 버튼
        btn_clear = QPushButton('[ CLEAR ALL ]')
        btn_clear.setFixedWidth(110)
        btn_clear.clicked.connect(scene.clear_targets)
        layout.addWidget(btn_clear)

        btn_demo = QPushButton('[ DEMO ]')
        btn_demo.setFixedWidth(80)
        btn_demo.clicked.connect(scene.add_default_targets)
        layout.addWidget(btn_demo)

        layout.addStretch()

        self._spn_vel.valueChanged.connect(self._on_vel_change)
        self._cmb_rcs.currentTextChanged.connect(self._on_rcs_change)
        scene.targets_changed.connect(self._on_targets_change)

        self._set_enabled(False)

    def _lbl(self, txt):
        l = QLabel(txt)
        l.setStyleSheet('color:#007a20; font-size:11px;')
        return l

    def _on_targets_change(self, _):
        self._refresh()

    def _refresh(self):
        tgt = self._scene.get_selected()
        self._set_enabled(tgt is not None)
        if tgt is None:
            return
        self._building = True
        self._spn_vel.setValue(tgt.velocity_mps)
        # RCS 콤보박스 동기화
        for i, (k, v) in enumerate(RCS_PRESETS.items()):
            if abs(v - tgt.rcs_m2) < 0.01:
                self._cmb_rcs.setCurrentIndex(i)
                break
        self._building = False

    def _set_enabled(self, en: bool):
        self._spn_vel.setEnabled(en)
        self._cmb_rcs.setEnabled(en)

    def _on_vel_change(self, v):
        if not self._building:
            self._scene.update_selected(velocity_mps=v)

    def _on_rcs_change(self, text):
        if not self._building:
            rcs = RCS_PRESETS.get(text, 1.0)
            self._scene.update_selected(rcs_m2=rcs)
