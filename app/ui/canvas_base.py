"""
matplotlib 임베딩 기반 클래스
모든 탭의 공통 스타일을 정의
"""
import sys
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── 한글 폰트 설정 (korean_mpl.py) ──
_root = Path(__file__).resolve().parents[2]   # FMCW/
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
try:
    from korean_mpl import configure_korean_font
    configure_korean_font()
except ImportError:
    pass

# ── 레이더 컬러 팔레트 ──
C = {
    'bg':          '#0d1117',
    'axes_bg':     '#060d06',
    'green':       '#00ff41',
    'green_mid':   '#00cc33',
    'green_dim':   '#007a20',
    'green_dark':  '#0d2a0d',
    'orange':      '#ff8800',
    'red':         '#ff2244',
    'cyan':        '#00ffee',
    'yellow':      '#ffe066',
    'white':       '#ccffcc',
    'cmap_rdm':    'Greens',      # Range-Doppler Map 컬러맵
    'cmap_signal': 'plasma',
}

MPLRC = {
    'figure.facecolor':   C['bg'],
    'axes.facecolor':     C['axes_bg'],
    'axes.edgecolor':     C['green_dark'],
    'axes.labelcolor':    C['green_dim'],
    'axes.titlecolor':    C['green'],
    'xtick.color':        C['green_dim'],
    'ytick.color':        C['green_dim'],
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'axes.labelsize':     11,
    'axes.titlesize':     12,
    'grid.color':         C['green_dark'],
    'grid.alpha':         0.7,
    'grid.linewidth':     0.4,
    'lines.color':        C['green'],
    'text.color':         C['green'],
    'legend.facecolor':   '#060d06',
    'legend.edgecolor':   C['green_dark'],
    'legend.labelcolor':  C['green_dim'],
    'legend.fontsize':    10,
    'figure.autolayout':  True,
}


class RadarCanvas(FigureCanvas):
    """
    FigureCanvas 기반 레이더 스타일 캔버스
    서브클래스는 _build_axes() 와 update_plots() 를 구현
    """

    def __init__(self, nrows: int = 1, ncols: int = 1,
                 figsize=None, tight=True):
        import matplotlib as mpl
        for k, v in MPLRC.items():
            mpl.rcParams[k] = v

        self.fig = Figure(figsize=figsize or (8, 4),
                          facecolor=C['bg'],
                          tight_layout=tight)
        super().__init__(self.fig)
        self.setStyleSheet(f"background-color: {C['bg']}; border: none;")
        self.axes = []
        self._build_axes(nrows, ncols)

    def _build_axes(self, nrows, ncols):
        """기본 axes 생성 — 서브클래스에서 override 가능"""
        for i in range(nrows * ncols):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            self._style_ax(ax)
            self.axes.append(ax)

    def _style_ax(self, ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor(C['axes_bg'])
        ax.tick_params(colors=C['green_dim'], length=3, width=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(C['green_dark'])
            spine.set_linewidth(0.8)
        ax.grid(True, color=C['green_dark'], alpha=0.7, linewidth=0.4)
        if title:
            ax.set_title(title, color=C['green'], fontsize=9, pad=4)
        if xlabel:
            ax.set_xlabel(xlabel, color=C['green_dim'], fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color=C['green_dim'], fontsize=8)

    def redraw(self):
        self.fig.canvas.draw_idle()

    # ── 공통 그리기 헬퍼 ──
    def _plot_range_profile(self, ax, range_ax, profile_db,
                            threshold_db=None, detections=None,
                            title='Range Profile'):
        ax.cla()
        self._style_ax(ax, title=title, xlabel='Range (m)', ylabel='dB')

        ax.plot(range_ax, profile_db, color=C['green'], linewidth=0.8,
                label='Signal')

        if threshold_db is not None and len(threshold_db) == len(range_ax):
            ax.plot(range_ax, threshold_db, color=C['orange'],
                    linewidth=0.8, linestyle='--', label='CFAR Threshold')

        if detections is not None:
            # detections: bool 1D over range bins (any doppler)
            det_r = np.any(detections, axis=0) if detections.ndim == 2 else detections
            det_idx = np.where(det_r)[0]
            if len(det_idx):
                ax.scatter(range_ax[det_idx], profile_db[det_idx],
                           color=C['red'], s=20, zorder=5, label='Detection')

        ax.legend(loc='upper right')

    def _plot_rdm(self, ax, rdm_db, range_ax, vel_ax,
                  detections=None, title='Range-Doppler Map',
                  vmin=None, vmax=None):
        ax.cla()
        self._style_ax(ax, title=title,
                       xlabel='Range (m)', ylabel='Velocity (m/s)')

        if vmin is None:
            vmin = np.percentile(rdm_db, 5)
        if vmax is None:
            vmax = rdm_db.max()

        r_min, r_max = float(range_ax[0]), float(range_ax[-1])
        v_min, v_max = float(vel_ax[-1]), float(vel_ax[0])  # fftshift: 위쪽이 음수

        im = ax.imshow(
            rdm_db,
            aspect='auto',
            origin='upper',
            extent=[r_min, r_max, v_max, v_min],
            cmap='Greens',
            vmin=vmin, vmax=vmax,
        )

        if detections is not None and detections.any():
            di, ri = np.where(detections)
            r_pts = range_ax[np.clip(ri, 0, len(range_ax) - 1)]
            v_pts = vel_ax[np.clip(di, 0, len(vel_ax) - 1)]
            ax.scatter(r_pts, v_pts, marker='x',
                       color=C['red'], s=30, linewidths=1.0, zorder=5,
                       label='Detection')
            ax.legend(loc='upper right')

        return im
