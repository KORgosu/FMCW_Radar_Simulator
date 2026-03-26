"""
FMCW 레이더 신호처리 엔진
모든 중간 결과를 SimResult 딕셔너리로 반환
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.ndimage import uniform_filter

from .models import RadarParams, Target


# ─────────────────────────────────────────────
#  결과 컨테이너
# ─────────────────────────────────────────────
@dataclass
class SimResult:
    # ── 파형 시각화용 ──
    t_fast: np.ndarray = field(default_factory=lambda: np.array([]))
    tx_chirp: np.ndarray = field(default_factory=lambda: np.array([]))   # 복소 처프
    tx_freq: np.ndarray = field(default_factory=lambda: np.array([]))    # 순간 주파수
    beat_one: np.ndarray = field(default_factory=lambda: np.array([]))   # 첫 처프 beat

    # ── Range-FFT ──
    range_axis: np.ndarray = field(default_factory=lambda: np.array([]))
    range_profile_db: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Beat Matrix (single-channel) ──
    beat_matrix: np.ndarray = field(default_factory=lambda: np.array([]))   # [N_chirp × N_sample]

    # ── Range-Doppler Map ──
    velocity_axis: np.ndarray = field(default_factory=lambda: np.array([]))
    rdm_db: np.ndarray = field(default_factory=lambda: np.array([]))         # [N_doppler × N_range]
    rdm_power: np.ndarray = field(default_factory=lambda: np.array([]))      # 선형 파워 (CFAR용)

    # ── CFAR ──
    cfar_threshold_db: np.ndarray = field(default_factory=lambda: np.array([]))
    cfar_detections: np.ndarray = field(default_factory=lambda: np.array([]))  # bool [N_d × N_r]

    # ── MIMO / AoA ──
    angle_axis: np.ndarray = field(default_factory=lambda: np.array([]))
    rdm_cube: np.ndarray = field(default_factory=lambda: np.array([]))  # [N_d × N_r × N_virtual]
    point_cloud: List[Dict] = field(default_factory=list)               # [{range, velocity, angle, power}]


# ─────────────────────────────────────────────
#  시뮬레이터
# ─────────────────────────────────────────────
class FMCWSimulator:
    """
    FMCW 레이더 신호처리 전체 파이프라인
    compute() 호출 → SimResult 반환
    """

    C = 3e8  # 광속

    def compute(self, targets: List[Target], params: RadarParams) -> SimResult:
        res = SimResult()

        if not targets:
            # 타겟 없을 때 빈 결과
            self._fill_empty(res, params)
            return res

        # 1. 기본 파형 정보
        self._compute_waveform(res, params)

        # 2. Beat matrix [N_chirp × N_sample]  (single virtual channel, index 0)
        beat_cube = self._generate_beat_cube(targets, params)    # [N_c × N_s × N_v]
        res.beat_matrix = beat_cube[:, :, 0]

        # 3. Range-FFT (첫 처프로 beat_one, full matrix로 range profile)
        res.beat_one = res.beat_matrix[0]
        self._compute_range_profile(res, params)

        # 4. Range-Doppler Map (single channel)
        rdm_complex = self._rdm(res.beat_matrix, params)          # [N_d × N_r]
        res.rdm_power = np.abs(rdm_complex) ** 2
        res.rdm_db = 20 * np.log10(np.abs(rdm_complex) + 1e-12)
        res.velocity_axis = params.velocity_axis()
        res.range_axis = params.range_axis()

        # 5. 2D CA-CFAR
        self._cfar(res, params)

        # 6. MIMO RDM cube + AoA
        res.rdm_cube = self._rdm_cube(beat_cube, params)          # [N_d × N_r × N_v]
        res.angle_axis = params.angle_axis()
        self._aoa_point_cloud(res, params)

        return res

    # ─────────────────────────────
    #  내부 처리 단계
    # ─────────────────────────────
    def _fill_empty(self, res: SimResult, p: RadarParams):
        N = p.n_sample
        res.t_fast = np.linspace(0, p.chirp_dur_s, N)
        res.tx_chirp = np.exp(1j * np.pi * p.mu * res.t_fast ** 2)
        res.tx_freq = p.mu * res.t_fast
        res.beat_one = np.zeros(N, dtype=complex)
        res.range_axis = p.range_axis()
        res.range_profile_db = np.full(p.n_range_fft // 2, -80.0)
        res.velocity_axis = p.velocity_axis()
        n_d = p.n_doppler_fft
        n_r = p.n_range_fft // 2
        res.rdm_db = np.full((n_d, n_r), -80.0)
        res.rdm_power = np.zeros((n_d, n_r))
        res.cfar_threshold_db = np.full(n_r, -80.0)
        res.cfar_detections = np.zeros((n_d, n_r), dtype=bool)
        res.beat_matrix = np.zeros((p.n_chirp, N), dtype=complex)
        res.angle_axis = p.angle_axis()
        res.rdm_cube = np.zeros((n_d, n_r, p.n_virtual), dtype=complex)
        res.point_cloud = []

    def _compute_waveform(self, res: SimResult, p: RadarParams):
        N = p.n_sample
        t = np.linspace(0, p.chirp_dur_s, N)
        res.t_fast = t
        res.tx_chirp = np.exp(1j * np.pi * p.mu * t ** 2)
        res.tx_freq = p.mu * t   # 순간 주파수 (베이스밴드 기준)

    def _generate_beat_cube(
        self, targets: List[Target], p: RadarParams
    ) -> np.ndarray:
        """
        Returns beat_cube [N_chirp × N_sample × N_virtual]
        모든 타겟의 beat signal을 합산 + 복소 가우시안 잡음 추가
        """
        N_c, N_s, N_v = p.n_chirp, p.n_sample, p.n_virtual
        t = np.linspace(0, p.chirp_dur_s, N_s)   # fast time [N_s]
        n_idx = np.arange(N_c, dtype=float)        # chirp index [N_c]

        # TX 위치 (lambda/2 간격 × n_rx)
        tx_pos = np.arange(p.n_tx) * p.n_rx * p.d_elem    # [N_tx]
        rx_pos = np.arange(p.n_rx) * p.d_elem              # [N_rx]
        # 가상 배열 위치: TX-RX 쌍별 합산 → [N_virtual]
        va_pos = np.array([tx + rx for tx in tx_pos for rx in rx_pos])

        beat_cube = np.zeros((N_c, N_s, N_v), dtype=complex)

        for tgt in targets:
            R0 = tgt.range_m
            v = tgt.velocity_mps
            theta_rad = np.radians(tgt.angle_deg)
            A = tgt.amplitude

            # 각 처프 n에서의 거리
            R_n = R0 + v * n_idx * p.t_pri   # [N_c]

            # beat frequency per chirp
            fb_n = 2.0 * R_n * p.mu / self.C  # [N_c]

            # 초기 위상 per chirp (slow-time Doppler)
            phi_0_n = -4.0 * np.pi * R_n / p.lam   # [N_c]

            # beat signal [N_c × N_s] (브로드캐스트)
            beat_2d = A * np.exp(
                1j * (2.0 * np.pi * fb_n[:, np.newaxis] * t[np.newaxis, :]
                      + phi_0_n[:, np.newaxis])
            )

            # 공간 위상 per virtual element [N_v]
            phase_spatial = 2.0 * np.pi / p.lam * va_pos * np.sin(theta_rad)

            # MIMO 확장 [N_c × N_s × N_v]
            beat_cube += (beat_2d[:, :, np.newaxis]
                          * np.exp(1j * phase_spatial)[np.newaxis, np.newaxis, :])

        # 복소 가우시안 잡음
        noise = (np.random.normal(0, p.noise_std, beat_cube.shape)
                 + 1j * np.random.normal(0, p.noise_std, beat_cube.shape))
        beat_cube += noise

        return beat_cube

    def _compute_range_profile(self, res: SimResult, p: RadarParams):
        """beat_matrix → 평균 Range profile (dB)"""
        win = np.hanning(p.n_sample)
        windowed = res.beat_matrix * win[np.newaxis, :]
        rng_fft = np.fft.fft(windowed, n=p.n_range_fft, axis=1)
        # 양의 주파수 절반만 (실수 신호와 달리 음의 주파수도 의미 있지만 Range용)
        N_half = p.n_range_fft // 2
        profile = np.mean(np.abs(rng_fft[:, :N_half]), axis=0)
        res.range_profile_db = 20 * np.log10(profile + 1e-12)

    def _rdm(self, beat_matrix: np.ndarray, p: RadarParams) -> np.ndarray:
        """
        2D-FFT → Range-Doppler Map (복소수)
        Returns [N_doppler × N_range/2]
        """
        N_half = p.n_range_fft // 2

        # Window 적용
        win_fast = np.hanning(p.n_sample)
        win_slow = np.hanning(p.n_chirp)

        mat = beat_matrix * win_fast[np.newaxis, :]

        # 1st FFT: Range (axis=1)
        rng_fft = np.fft.fft(mat, n=p.n_range_fft, axis=1)[:, :N_half]

        # Doppler window 적용
        rng_fft *= win_slow[:, np.newaxis]

        # 2nd FFT: Doppler (axis=0) + fftshift
        rdm = np.fft.fftshift(
            np.fft.fft(rng_fft, n=p.n_doppler_fft, axis=0),
            axes=0
        )
        return rdm  # [N_doppler × N_range/2]

    def _cfar(self, res: SimResult, p: RadarParams):
        """2D CA-CFAR (scipy uniform_filter 기반 고속 구현)"""
        power = res.rdm_power   # [N_d × N_r]
        ng, nt = p.cfar_guard, p.cfar_train
        pfa = p.cfar_pfa

        outer = 2 * (ng + nt) + 1
        inner = 2 * ng + 1
        N_train = outer ** 2 - inner ** 2
        alpha = N_train * (pfa ** (-1.0 / N_train) - 1.0)

        # 박스 필터로 각 영역 합산 근사
        sum_outer = uniform_filter(power, size=outer, mode='wrap') * outer ** 2
        sum_inner = uniform_filter(power, size=inner, mode='wrap') * inner ** 2

        noise_mean = np.maximum((sum_outer - sum_inner) / N_train, 1e-30)
        threshold_power = alpha * noise_mean

        res.cfar_detections = power > threshold_power
        # Range profile용 1D threshold (Doppler 축 평균)
        thr_1d = np.mean(threshold_power, axis=0)
        res.cfar_threshold_db = 20 * np.log10(thr_1d + 1e-12)

    def _rdm_cube(self, beat_cube: np.ndarray, p: RadarParams) -> np.ndarray:
        """
        MIMO beat cube [N_c × N_s × N_v] → RDM cube [N_d × N_r/2 × N_v]
        """
        N_v = p.n_virtual
        N_half = p.n_range_fft // 2
        rdm_cube = np.zeros((p.n_doppler_fft, N_half, N_v), dtype=complex)

        for m in range(N_v):
            rdm_cube[:, :, m] = self._rdm(beat_cube[:, :, m], p)

        return rdm_cube

    def _aoa_point_cloud(self, res: SimResult, p: RadarParams):
        """
        탐지된 (Doppler bin, Range bin) 위치에서 3rd FFT → AoA 추정
        → point_cloud: [{range, velocity, angle, power_db}]
        """
        det_idx = np.argwhere(res.cfar_detections)   # [[d_i, r_i], ...]
        points = []

        angle_ax = res.angle_axis   # [N_angle_fft]

        for (di, ri) in det_idx:
            # 해당 cell의 가상 배열 신호 [N_v]
            va_signal = res.rdm_cube[di, ri, :]

            # 3rd FFT: AoA
            angle_fft = np.fft.fftshift(
                np.fft.fft(va_signal, n=p.n_angle_fft)
            )
            magnitude = np.abs(angle_fft)

            peak_idx = int(np.argmax(magnitude))
            angle_est = float(angle_ax[peak_idx])
            power_db = float(res.rdm_db[di, ri])

            range_est = float(res.range_axis[ri]) if ri < len(res.range_axis) else 0.0
            vel_est = float(res.velocity_axis[di]) if di < len(res.velocity_axis) else 0.0

            points.append({
                'range': range_est,
                'velocity': vel_est,
                'angle': angle_est,
                'power_db': power_db,
            })

        res.point_cloud = points
