"""
Tab 1 — 신호 원리
  ┌──────────────────────┬──────────────────────┐
  │  TX Chirp 주파수 스윕 │  TX vs RX 주파수 비교 │
  ├──────────────────────┼──────────────────────┤
  │  Beat Signal (time)  │  Range Profile (FFT)  │
  └──────────────────────┴──────────────────────┘
"""
from __future__ import annotations
import numpy as np

from PyQt5.QtWidgets import QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

from ..canvas_base import RadarCanvas, C as COL
from ...core.models import RadarParams, Target
from ...core.simulator import SimResult


class Tab1Canvas(RadarCanvas):
    def __init__(self):
        super().__init__(figsize=(10, 6))
        self.fig.subplots_adjust(
            left=0.07, right=0.97, top=0.93, bottom=0.09,
            wspace=0.35, hspace=0.45
        )

    def _build_axes(self, nrows, ncols):
        self.ax_chirp  = self.fig.add_subplot(2, 2, 1)
        self.ax_txrx   = self.fig.add_subplot(2, 2, 2)
        self.ax_beat   = self.fig.add_subplot(2, 2, 3)
        self.ax_range  = self.fig.add_subplot(2, 2, 4)
        for ax in [self.ax_chirp, self.ax_txrx, self.ax_beat, self.ax_range]:
            self._style_ax(ax)

    def update_plots(self, res: SimResult, params: RadarParams,
                     targets: list):
        self._draw_chirp(res, params)
        self._draw_txrx(res, params, targets)
        self._draw_beat(res, params)
        self._draw_range(res, params)
        self.redraw()

    # ── 서브플롯 1: TX Chirp 순간 주파수 ──
    def _draw_chirp(self, res: SimResult, p: RadarParams):
        ax = self.ax_chirp
        ax.cla()
        self._style_ax(ax,
                       title='TX Chirp — 순간 주파수',
                       xlabel='Time (μs)', ylabel='Freq (MHz)')
        t_us = res.t_fast * 1e6
        freq_mhz = res.tx_freq / 1e6
        ax.plot(t_us, freq_mhz, color=COL['green'], linewidth=1.2)
        ax.fill_between(t_us, freq_mhz, alpha=0.08, color=COL['green'])
        ax.set_xlim(0, p.chirp_dur_s * 1e6)
        ax.set_ylim(0, p.bandwidth_hz / 1e6 * 1.05)
        # 대역폭 화살표 레이블
        ax.annotate('', xy=(t_us[-1], freq_mhz[-1]),
                    xytext=(t_us[-1], 0),
                    arrowprops=dict(arrowstyle='<->', color=COL['orange'],
                                    lw=1.0))
        ax.text(t_us[-1] * 0.85, freq_mhz[-1] / 2,
                f'B={p.bandwidth_hz/1e6:.0f}MHz',
                color=COL['orange'], fontsize=10)

    # ── 서브플롯 2: TX vs RX 주파수 비교 ──
    def _draw_txrx(self, res: SimResult, p: RadarParams, targets: list):
        ax = self.ax_txrx
        ax.cla()
        self._style_ax(ax,
                       title='TX vs RX — Beat Frequency',
                       xlabel='Time (μs)', ylabel='Freq (MHz)')
        t_us = res.t_fast * 1e6
        tx_f = res.tx_freq / 1e6

        ax.plot(t_us, tx_f, color=COL['green'], linewidth=1.2,
                label='TX', zorder=3)

        colors = [COL['cyan'], COL['yellow'], COL['orange'], COL['red']]
        for i, tgt in enumerate(targets[:4]):
            tau = 2 * tgt.range_m / 3e8
            fb = 2 * tgt.range_m * p.mu / 3e8
            rx_f = tx_f - fb / 1e6   # 지연 → 주파수 낮음
            col = colors[i % len(colors)]
            ax.plot(t_us, rx_f, color=col, linewidth=1.0,
                    linestyle='--', label=f'T{i+1} (R={tgt.range_m:.0f}m)',
                    zorder=2)
            # fb 표시
            ax.annotate(
                f'fb={fb/1e3:.1f}kHz',
                xy=(t_us[len(t_us)//2], (tx_f[len(t_us)//2] + rx_f[len(t_us)//2]) / 2),
                fontsize=9, color=col
            )

        ax.set_xlim(0, p.chirp_dur_s * 1e6)
        ax.legend(fontsize=9, loc='upper left')

    # ── 서브플롯 3: Beat Signal (time domain) ──
    def _draw_beat(self, res: SimResult, p: RadarParams):
        ax = self.ax_beat
        ax.cla()
        self._style_ax(ax,
                       title='Beat Signal (첫 번째 Chirp)',
                       xlabel='Time (μs)', ylabel='Amplitude')
        t_us = res.t_fast * 1e6
        ax.plot(t_us, res.beat_one.real, color=COL['green'],
                linewidth=0.7, label='Re')
        ax.plot(t_us, res.beat_one.imag, color=COL['cyan'],
                linewidth=0.7, alpha=0.6, label='Im')
        ax.set_xlim(0, p.chirp_dur_s * 1e6)
        ax.legend(fontsize=9, loc='upper right')

    # ── 서브플롯 4: Range Profile (FFT) ──
    def _draw_range(self, res: SimResult, p: RadarParams):
        ax = self.ax_range
        ax.cla()
        self._style_ax(ax,
                       title='Range Profile (Range-FFT)',
                       xlabel='Range (m)', ylabel='dB')
        if len(res.range_axis) == 0 or len(res.range_profile_db) == 0:
            return
        ax.plot(res.range_axis, res.range_profile_db,
                color=COL['green'], linewidth=0.9)
        ax.set_xlim(0, p.range_max_m)
        # ΔR 레이블
        ax.text(0.02, 0.05, f'ΔR = {p.range_resolution:.2f} m',
                transform=ax.transAxes, color=COL['green_dim'], fontsize=10)


class Tab1Widget(Tab1Canvas):
    """Tab1 전체 위젯 (캔버스 = 위젯)"""
    pass
