"""
Tab 2 — Range-Doppler Map
  ┌──────────────┬──────────────┬──────────────────────────────┐
  │ Beat Matrix  │ Range Profile │      Range-Doppler Map        │
  │  heatmap     │  (평균)       │      (main, 크게)             │
  └──────────────┴──────────────┴──────────────────────────────┘
"""
from __future__ import annotations
import numpy as np

from ..canvas_base import RadarCanvas, C as COL
from ...core.models import RadarParams
from ...core.simulator import SimResult


class Tab2Canvas(RadarCanvas):
    def __init__(self):
        super().__init__(figsize=(11, 5))
        self.fig.subplots_adjust(
            left=0.06, right=0.97, top=0.92, bottom=0.11,
            wspace=0.38
        )
        self._cb = None   # colorbar 참조 보관

    def _build_axes(self, nrows, ncols):
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 2])
        self.ax_beat  = self.fig.add_subplot(gs[0])
        self.ax_range = self.fig.add_subplot(gs[1])
        self.ax_rdm   = self.fig.add_subplot(gs[2])
        for ax in [self.ax_beat, self.ax_range, self.ax_rdm]:
            self._style_ax(ax)

    def update_plots(self, res: SimResult, params: RadarParams, targets: list):
        self._draw_beat_matrix(res, params)
        self._draw_range_profile(res, params)
        self._draw_rdm(res, params)
        self.redraw()

    def _draw_beat_matrix(self, res: SimResult, p: RadarParams):
        ax = self.ax_beat
        ax.cla()
        self._style_ax(ax,
                       title='Beat Matrix\n[N_chirp × N_sample]',
                       xlabel='Sample', ylabel='Chirp #')
        if res.beat_matrix.size == 0:
            return
        data = np.abs(res.beat_matrix)
        ax.imshow(data, aspect='auto', origin='lower',
                  cmap='Greens', interpolation='nearest')
        ax.text(0.02, 0.97, f'shape: {res.beat_matrix.shape}',
                transform=ax.transAxes, color='black',
                fontsize=9, va='top',
                bbox=dict(facecolor='white', alpha=0.6, pad=2, edgecolor='none'))

    def _draw_range_profile(self, res: SimResult, p: RadarParams):
        ax = self.ax_range
        ax.cla()
        self._style_ax(ax,
                       title='Range Profile\n(Range-FFT 평균)',
                       xlabel='Range (m)', ylabel='dB')
        if len(res.range_axis) == 0:
            return
        ax.plot(res.range_axis, res.range_profile_db,
                color=COL['green'], linewidth=0.9)
        ax.set_xlim(0, p.range_max_m)

        # 분해능 표시
        ax.text(0.02, 0.05,
                f'ΔR={p.range_resolution:.2f}m  Rmax={p.range_max_m:.0f}m',
                transform=ax.transAxes, color=COL['green_dim'], fontsize=9)

    def _draw_rdm(self, res: SimResult, p: RadarParams):
        ax = self.ax_rdm
        ax.cla()
        self._style_ax(ax,
                       title='Range-Doppler Map  (2D-FFT)',
                       xlabel='Range (m)', ylabel='Velocity (m/s)')
        if res.rdm_db.size == 0 or len(res.range_axis) == 0:
            return

        vmin = float(np.percentile(res.rdm_db, 5))
        vmax = float(res.rdm_db.max())

        r0, r1 = float(res.range_axis[0]),  float(res.range_axis[-1])
        v0, v1 = float(res.velocity_axis[-1]), float(res.velocity_axis[0])

        im = ax.imshow(
            res.rdm_db,
            aspect='auto', origin='upper',
            extent=[r0, r1, v0, v1],
            cmap='Greens', vmin=vmin, vmax=vmax,
        )

        if self._cb is None:
            self._cb = self.fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04)
            self._cb.ax.tick_params(labelsize=9, colors=COL['green_dim'])
            self._cb.set_label('dB', color=COL['green_dim'], fontsize=10)
        else:
            self._cb.update_normal(im)

        ax.axhline(0, color=COL['green_dark'], linewidth=0.6, linestyle='--')

        # 정보
        ax.text(0.01, 0.98,
                f'Vmax=±{p.velocity_max_mps:.1f}m/s  '
                f'ΔV={p.velocity_resolution:.2f}m/s',
                transform=ax.transAxes, color=COL['green_dim'],
                fontsize=9, va='top')


class Tab2Widget(Tab2Canvas):
    pass
