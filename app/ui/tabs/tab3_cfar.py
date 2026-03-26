"""
Tab 3 — CFAR 탐지
  ┌─────────────────────────────────────────────────────────┐
  │  Range Profile + CFAR 임계값 오버레이 (상단, 넓게)        │
  ├──────────────────────────┬──────────────────────────────┤
  │  R-D Map + Detection 마킹 │  CFAR 윈도우 구조 다이어그램  │
  └──────────────────────────┴──────────────────────────────┘
"""
from __future__ import annotations
import numpy as np
import matplotlib.patches as mpatches

from ..canvas_base import RadarCanvas, C as COL
from ...core.models import RadarParams
from ...core.simulator import SimResult


class Tab3Canvas(RadarCanvas):
    def __init__(self):
        super().__init__(figsize=(11, 6))
        self.fig.subplots_adjust(
            left=0.07, right=0.97, top=0.93, bottom=0.09,
            wspace=0.35, hspace=0.45
        )

    def _build_axes(self, nrows, ncols):
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
        self.ax_profile = self.fig.add_subplot(gs[0, :])   # 전체 너비
        self.ax_rdm     = self.fig.add_subplot(gs[1, 0])
        self.ax_window  = self.fig.add_subplot(gs[1, 1])
        for ax in [self.ax_profile, self.ax_rdm, self.ax_window]:
            self._style_ax(ax)

    def update_plots(self, res: SimResult, params: RadarParams, targets: list):
        self._draw_profile(res, params)
        self._draw_rdm(res, params)
        self._draw_cfar_window(res, params)
        self.redraw()

    def _draw_profile(self, res: SimResult, p: RadarParams):
        ax = self.ax_profile
        ax.cla()
        self._style_ax(ax,
                       title='Range Profile + CA-CFAR 임계값',
                       xlabel='Range (m)', ylabel='Power (dB)')
        if len(res.range_axis) == 0:
            return

        ax.plot(res.range_axis, res.range_profile_db,
                color=COL['green'], linewidth=0.9, label='Signal', zorder=3)

        if len(res.cfar_threshold_db) == len(res.range_axis):
            ax.plot(res.range_axis, res.cfar_threshold_db,
                    color=COL['orange'], linewidth=0.9,
                    linestyle='--', label='CFAR Threshold', zorder=2)

        # 탐지 마킹
        if res.cfar_detections.size > 0:
            det_1d = np.any(res.cfar_detections, axis=0)
            det_idx = np.where(det_1d)[0]
            if len(det_idx):
                ax.scatter(
                    res.range_axis[det_idx],
                    res.range_profile_db[det_idx],
                    color=COL['red'], s=25, zorder=5,
                    label=f'Detection ({len(det_idx)})'
                )

        ax.set_xlim(0, p.range_max_m)
        ax.legend(loc='upper right', fontsize=10)

        # Pfa 레이블
        ax.text(0.01, 0.96,
                f'Pfa={p.cfar_pfa:.0e}  Guard={p.cfar_guard}  '
                f'Train={p.cfar_train}',
                transform=ax.transAxes, color=COL['green_dim'],
                fontsize=10, va='top')

    def _draw_rdm(self, res: SimResult, p: RadarParams):
        ax = self.ax_rdm
        ax.cla()
        self._style_ax(ax,
                       title='R-D Map + Detection',
                       xlabel='Range (m)', ylabel='Velocity (m/s)')
        if res.rdm_db.size == 0:
            return

        r0, r1 = float(res.range_axis[0]), float(res.range_axis[-1])
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
            ax.scatter(r_pts, v_pts, marker='x',
                       color=COL['red'], s=35, linewidths=1.2, zorder=5)

        n_det = int(res.cfar_detections.sum())
        ax.text(0.02, 0.96, f'탐지: {n_det}개',
                transform=ax.transAxes, color=COL['red'],
                fontsize=10, va='top')

    def _draw_cfar_window(self, res: SimResult, p: RadarParams):
        """CFAR 슬라이딩 윈도우 구조 시각화"""
        ax = self.ax_window
        ax.cla()
        self._style_ax(ax, title='CA-CFAR 윈도우 구조')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')

        ng = p.cfar_guard
        nt = p.cfar_train
        total = ng + nt

        cell_w = 0.06
        gap = 0.01
        step = cell_w + gap
        n_show = min(total, 6)

        def draw_cells(x_start, count, color, label):
            for i in range(count):
                x = x_start + i * step
                rect = mpatches.FancyBboxPatch(
                    (x, -0.1), cell_w, 0.3,
                    boxstyle='round,pad=0.005',
                    facecolor=color, edgecolor=COL['green_dark'],
                    linewidth=0.5
                )
                ax.add_patch(rect)

        # 왼쪽 train
        x0 = -0.9
        draw_cells(x0, n_show, '#001a00', 'Train')
        # 왼쪽 guard
        x1 = x0 + n_show * step + gap
        draw_cells(x1, min(ng, 3), '#002a00', 'Guard')
        # CUT
        x2 = x1 + min(ng, 3) * step + gap
        rect_cut = mpatches.FancyBboxPatch(
            (x2, -0.12), cell_w * 1.3, 0.34,
            boxstyle='round,pad=0.005',
            facecolor='#004020', edgecolor=COL['green'],
            linewidth=1.2
        )
        ax.add_patch(rect_cut)
        # 오른쪽 guard
        x3 = x2 + cell_w * 1.3 + gap * 2
        draw_cells(x3, min(ng, 3), '#002a00', 'Guard')
        # 오른쪽 train
        x4 = x3 + min(ng, 3) * step + gap
        draw_cells(x4, n_show, '#001a00', 'Train')

        # 레이블
        def lbl(x, txt, col):
            ax.text(x, 0.3, txt, color=col, fontsize=10, ha='center', va='bottom')

        lbl(x0 + n_show * step / 2, f'Train\n(×{nt})', COL['green_dim'])
        lbl(x1 + min(ng,3)*step/2, f'Guard\n(×{ng})', COL['green_mid'])
        lbl(x2 + cell_w*0.65, 'CUT', COL['green'])
        lbl(x3 + min(ng,3)*step/2, f'Guard\n(×{ng})', COL['green_mid'])
        lbl(x4 + n_show*step/2, f'Train\n(×{nt})', COL['green_dim'])

        # 수식
        N = 2 * nt
        alpha = N * (p.cfar_pfa ** (-1/N) - 1)
        ax.text(0, -0.55,
                f'N_train = {2*nt}  α = {alpha:.2f}',
                color=COL['green_dim'], fontsize=11,
                ha='center', va='top')
        ax.text(0, -0.7,
                f'T = α × mean(train cells)',
                color=COL['green_mid'], fontsize=10,
                ha='center', va='top')
        ax.text(0, -0.85,
                f'Pfa = {p.cfar_pfa:.0e}  ->  alpha = N*(Pfa^(-1/N) - 1)',
                color=COL['green_dim'], fontsize=9,
                ha='center', va='top')


class Tab3Widget(Tab3Canvas):
    pass
