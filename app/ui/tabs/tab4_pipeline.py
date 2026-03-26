"""
Tab 4 — 통합 파이프라인 + Point Cloud
  ┌──────────────────────────┬──────────────────────────────┐
  │  R-D Map + Detection     │  2D Point Cloud (top-down)   │
  ├──────────────────────────┴──────────────────────────────┤
  │  처리 흐름 인디케이터 + 파라미터 요약                    │
  └─────────────────────────────────────────────────────────┘
"""
from __future__ import annotations
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

from ..canvas_base import RadarCanvas, C as COL
from ...core.models import RadarParams
from ...core.simulator import SimResult


# ─────────────────────────────────────────────
#  matplotlib 캔버스 (RDM + Point Cloud)
# ─────────────────────────────────────────────
class Tab4Canvas(RadarCanvas):
    def __init__(self):
        super().__init__(figsize=(11, 5))
        self.fig.subplots_adjust(
            left=0.07, right=0.97, top=0.92, bottom=0.11,
            wspace=0.35
        )

    def _build_axes(self, nrows, ncols):
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])
        self.ax_rdm   = self.fig.add_subplot(gs[0])
        self.ax_cloud = self.fig.add_subplot(gs[1])
        for ax in [self.ax_rdm, self.ax_cloud]:
            self._style_ax(ax)
        self._cb_cloud = None   # 속도 컬러바 참조

    def update_plots(self, res: SimResult, params: RadarParams, targets: list):
        self._draw_rdm(res, params)
        self._draw_point_cloud(res, params, targets)
        self.redraw()

    def _draw_rdm(self, res: SimResult, p: RadarParams):
        ax = self.ax_rdm
        ax.cla()
        self._style_ax(ax,
                       title='Range-Doppler Map + CFAR Detection',
                       xlabel='Range (m)', ylabel='Velocity (m/s)')
        if res.rdm_db.size == 0:
            return

        r0, r1 = float(res.range_axis[0]),  float(res.range_axis[-1])
        v0, v1 = float(res.velocity_axis[-1]), float(res.velocity_axis[0])
        vmin = float(np.percentile(res.rdm_db, 5))

        ax.imshow(
            res.rdm_db, aspect='auto', origin='upper',
            extent=[r0, r1, v0, v1],
            cmap='Greens', vmin=vmin, vmax=res.rdm_db.max()
        )

        if res.cfar_detections.any():
            di, ri = np.where(res.cfar_detections)
            r_pts = res.range_axis[np.clip(ri, 0, len(res.range_axis)-1)]
            v_pts = res.velocity_axis[np.clip(di, 0, len(res.velocity_axis)-1)]
            ax.scatter(r_pts, v_pts, marker='o', facecolors='none',
                       edgecolors=COL['red'], s=50, linewidths=1.2,
                       zorder=5, label='Detection')
            ax.legend(fontsize=10, loc='upper right')

        ax.axhline(0, color=COL['green_dark'], lw=0.6, linestyle='--')

    def _draw_point_cloud(self, res: SimResult, p: RadarParams, targets: list):
        ax = self.ax_cloud
        ax.cla()
        self._style_ax(ax,
                       title='Point Cloud (Top-Down View)',
                       xlabel='Cross-Range (m)', ylabel='Range (m)')
        ax.set_facecolor(COL['axes_bg'])

        r_max = p.range_max_m

        # 거리 링 (배경 장식)
        theta = np.linspace(-np.pi/2, np.pi/2, 200)
        for r_ring in range(50, int(r_max)+1, 50):
            ax.plot(r_ring * np.sin(theta), r_ring * np.cos(theta),
                    color=COL['green_dark'], linewidth=0.4, zorder=1)

        # Ground truth 타겟 (반투명)
        for tgt in targets:
            ang_r = np.radians(tgt.angle_deg)
            x_gt = tgt.range_m * np.sin(ang_r)
            y_gt = tgt.range_m * np.cos(ang_r)
            ax.scatter(x_gt, y_gt, marker='+',
                       color=COL['green_dim'], s=60, linewidths=1.0,
                       zorder=2, label='True' if tgt == targets[0] else '')

        # 탐지된 Point Cloud
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # 속도 컬러맵: 빨강(접근, +) → 흰색(정지, 0) → 파랑(이탈, -)
        cmap = cm.RdBu_r
        v_abs_max = p.velocity_max_mps or 1.0
        norm = mcolors.Normalize(vmin=-v_abs_max, vmax=v_abs_max)

        if res.point_cloud:
            r_pts = np.array([pt['range']    for pt in res.point_cloud])
            ang_r = np.radians([pt['angle']  for pt in res.point_cloud])
            v_pts = np.array([pt['velocity'] for pt in res.point_cloud])
            pwr   = np.array([pt['power_db'] for pt in res.point_cloud])

            x_pts = r_pts * np.sin(ang_r)
            y_pts = r_pts * np.cos(ang_r)

            colors = cmap(norm(v_pts))   # RGBA 배열
            sizes  = np.clip((pwr + 80) * 2, 40, 300)

            sc = ax.scatter(x_pts, y_pts, c=colors, s=sizes,
                            zorder=5, alpha=0.92, marker='o',
                            edgecolors='white', linewidths=0.4)

            for xi, yi, vi in zip(x_pts, y_pts, v_pts):
                ax.annotate(
                    f'{vi:+.1f}',
                    (xi, yi), xytext=(4, 4), textcoords='offset points',
                    fontsize=8, color='white',
                    bbox=dict(facecolor='#111111', alpha=0.5,
                              pad=1, edgecolor='none')
                )

        # 컬러바 범례 (속도 축) — 처음 한 번만 생성, 이후 update_normal
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if self._cb_cloud is None:
            self._cb_cloud = self.fig.colorbar(sm, ax=ax,
                                               pad=0.02, fraction=0.04)
            self._cb_cloud.ax.tick_params(labelsize=8, colors=COL['green_dim'])
            self._cb_cloud.set_label('Velocity (m/s)',
                                     color=COL['green_dim'], fontsize=9)
            self._cb_cloud.ax.text(0.5, 1.04, '접근(+)',
                                   transform=self._cb_cloud.ax.transAxes,
                                   ha='center', fontsize=8, color='#dd3333')
            self._cb_cloud.ax.text(0.5, -0.08, '이탈(-)',
                                   transform=self._cb_cloud.ax.transAxes,
                                   ha='center', fontsize=8, color='#3355dd')
        else:
            self._cb_cloud.update_normal(sm)

        # 레이더 아이콘
        ax.scatter(0, 0, marker='^', color=COL['green'], s=100, zorder=10)
        ax.text(1, -5, 'RADAR', color=COL['green_dim'], fontsize=9)

        ax.set_xlim(-r_max * 0.7, r_max * 0.7)
        ax.set_ylim(-10, r_max * 1.05)

        n_det = len(res.point_cloud)
        ax.text(0.02, 0.98,
                f'탐지: {n_det}개',
                transform=ax.transAxes, color=COL['green_dim'],
                fontsize=9, va='top')


# ─────────────────────────────────────────────
#  처리 흐름 인디케이터 (하단 바)
# ─────────────────────────────────────────────
class PipelineBar(QWidget):
    """Chirp → Beat → Range-FFT → Doppler-FFT → CFAR → AoA 흐름"""

    STAGES = ['CHIRP', 'BEAT\nMATRIX', 'RANGE\nFFT', 'DOPPLER\nFFT',
              'CFAR\nDETECT', 'AoA\nESTIM']

    # ── 상태별 스타일 ──
    _STYLE_DONE = (
        "background:#003a10; color:#00ff41; "
        "border:2px solid #00cc33; border-radius:4px; "
        "font-size:10px; font-weight:bold; padding:2px;"
    )
    _STYLE_ACTIVE = (
        "background:#005a20; color:#ffffff; "
        "border:2px solid #00ff41; border-radius:4px; "
        "font-size:10px; font-weight:bold; padding:2px;"
    )
    _STYLE_PENDING = (
        "background:#101010; color:#333333; "
        "border:1px dashed #2a2a2a; border-radius:4px; "
        "font-size:10px; padding:2px;"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self._current = -1

        # 단계별 애니메이션 타이머
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(120)
        self._anim_timer.timeout.connect(self._anim_step)
        self._anim_target = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self._lbls = []
        self._arrows = []
        for i, name in enumerate(self.STAGES):
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedWidth(80)
            lbl.setFixedHeight(52)
            lbl.setStyleSheet(self._STYLE_PENDING)
            layout.addWidget(lbl)
            self._lbls.append(lbl)

            if i < len(self.STAGES) - 1:
                arr = QLabel('→')
                arr.setAlignment(Qt.AlignCenter)
                arr.setStyleSheet("color:#222222; font-size:13px;")
                arr.setFixedWidth(18)
                layout.addWidget(arr)
                self._arrows.append(arr)

        layout.addStretch()

        self._info = QLabel('')
        self._info.setWordWrap(False)
        self._info.setStyleSheet(
            "color:#00aa30; font-size:11px; "
            "background:#040a04; border:1px solid #0d2a0d; "
            "padding:4px; letter-spacing:1px;"
        )
        layout.addWidget(self._info)

        # 초기 상태: 전부 미완료
        self._apply_all(self._STYLE_PENDING, arrow_color='#222222')

    def _apply_all(self, style, arrow_color='#222222'):
        for lbl in self._lbls:
            lbl.setStyleSheet(style)
        for arr in self._arrows:
            arr.setStyleSheet(f"color:{arrow_color}; font-size:13px;")

    def _apply_state(self, idx: int):
        """idx 단계까지 완료, idx+1이 현재 진행 중, 나머지는 미완료"""
        for i, lbl in enumerate(self._lbls):
            if i < idx:
                # 완료: 체크 접두어 + 밝은 초록
                base = self.STAGES[i].replace('\n', ' ')
                lbl.setText(f'✓ {base}')
                lbl.setStyleSheet(self._STYLE_DONE)
            elif i == idx:
                # 현재 처리 중: 흰 글씨 + 가장 밝은 테두리
                lbl.setText(self.STAGES[i])
                lbl.setStyleSheet(self._STYLE_ACTIVE)
            else:
                # 미완료: 어두운 회색
                lbl.setText(self.STAGES[i])
                lbl.setStyleSheet(self._STYLE_PENDING)

        # 화살표: 양쪽이 모두 완료된 것만 밝게
        for j, arr in enumerate(self._arrows):
            if j < idx - 1:
                arr.setStyleSheet("color:#00cc33; font-size:13px;")
            else:
                arr.setStyleSheet("color:#222222; font-size:13px;")

    def _anim_step(self):
        self._current += 1
        if self._current >= self._anim_target:
            # 마지막 단계: 전부 완료 표시
            for i, lbl in enumerate(self._lbls):
                base = self.STAGES[i].replace('\n', ' ')
                lbl.setText(f'✓ {base}')
                lbl.setStyleSheet(self._STYLE_DONE)
            for arr in self._arrows:
                arr.setStyleSheet("color:#00cc33; font-size:13px;")
            self._anim_timer.stop()
        else:
            self._apply_state(self._current)

    def animate(self):
        """단계 0부터 순서대로 켜지는 애니메이션 시작"""
        self._current = -1
        self._anim_target = len(self.STAGES)
        # 먼저 전부 초기화
        self._apply_all(self._STYLE_PENDING)
        self._anim_timer.start()

    def update_info(self, params: RadarParams, n_det: int):
        self._info.setText(
            f"ΔR={params.range_resolution:.2f}m  "
            f"Rmax={params.range_max_m:.0f}m  "
            f"Vmax=±{params.velocity_max_mps:.1f}m/s  "
            f"ΔV={params.velocity_resolution:.2f}m/s  "
            f"N_virtual={params.n_virtual}  "
            f"탐지={n_det}개"
        )


# ─────────────────────────────────────────────
#  Tab4 전체 위젯
# ─────────────────────────────────────────────
class Tab4Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._canvas = Tab4Canvas()
        layout.addWidget(self._canvas)

        self._bar = PipelineBar()
        layout.addWidget(self._bar)

    def update_plots(self, res: SimResult, params: RadarParams, targets: list):
        self._bar.animate()
        self._bar.update_info(params, len(res.point_cloud))
        self._canvas.update_plots(res, params, targets)
