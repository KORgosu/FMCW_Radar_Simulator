"""
RadarParams, Target 데이터 모델
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Target:
    range_m: float = 50.0       # 레이더로부터의 거리 (m)
    velocity_mps: float = 0.0   # 시선 방향 속도 (m/s, 양수 = 접근)
    angle_deg: float = 0.0      # 정면 기준 각도 (도)
    rcs_m2: float = 1.0         # 레이더 반사 단면적 (m²)

    @property
    def amplitude(self) -> float:
        """수신 신호 진폭 (√RCS / R²)"""
        return np.sqrt(max(self.rcs_m2, 1e-6)) / max(self.range_m, 1.0) ** 2

    def __str__(self):
        sign = "+" if self.velocity_mps >= 0 else ""
        return (f"R={self.range_m:.1f}m  "
                f"V={sign}{self.velocity_mps:.1f}m/s  "
                f"θ={self.angle_deg:+.1f}°  "
                f"RCS={self.rcs_m2:.1f}m²")


@dataclass
class RadarParams:
    # ── 파형 파라미터 ──
    fc_hz: float = 77e9          # 반송파 주파수 (Hz)
    bandwidth_hz: float = 150e6  # 처프 대역폭 (Hz)
    chirp_dur_s: float = 40e-6   # 처프 지속 시간 (s)
    prf_hz: float = 1000.0       # 펄스 반복 주파수 (Hz)
    fs_hz: float = 15e6          # ADC 샘플링 레이트 (Hz)
    n_chirp: int = 64            # 처프 수 (slow-time 샘플 수)

    # ── MIMO ──
    n_tx: int = 2                # TX 안테나 수
    n_rx: int = 4                # RX 안테나 수

    # ── 잡음 ──
    noise_std: float = 2e-5      # 잡음 표준편차 (복소 잡음 one-side)

    # ── CFAR ──
    cfar_pfa: float = 1e-4       # 오경보 확률
    cfar_guard: int = 2          # guard cell 수 (한쪽)
    cfar_train: int = 8          # training cell 수 (한쪽)

    # ── FFT 포인트 수 ──
    n_range_fft: int = 512
    n_doppler_fft: int = 128
    n_angle_fft: int = 256

    # ─────────── 파생 속성 ───────────
    @property
    def lam(self) -> float:
        return 3e8 / self.fc_hz

    @property
    def mu(self) -> float:
        """처프 레이트 (Hz/s)"""
        return self.bandwidth_hz / self.chirp_dur_s

    @property
    def t_pri(self) -> float:
        """처프 반복 주기 (s)"""
        return 1.0 / self.prf_hz

    @property
    def n_sample(self) -> int:
        """처프 당 ADC 샘플 수"""
        return int(self.fs_hz * self.chirp_dur_s)

    @property
    def n_virtual(self) -> int:
        return self.n_tx * self.n_rx

    @property
    def d_elem(self) -> float:
        """가상 배열 소자 간격 (λ/2)"""
        return self.lam / 2.0

    @property
    def range_resolution(self) -> float:
        return 3e8 / (2.0 * self.bandwidth_hz)

    @property
    def range_max_m(self) -> float:
        """ADC 샘플링 조건에서의 최대 탐지 거리"""
        return self.fs_hz * 3e8 / (4.0 * self.mu)

    @property
    def velocity_max_mps(self) -> float:
        return self.lam * self.prf_hz / 4.0

    @property
    def velocity_resolution(self) -> float:
        return self.lam / (2.0 * self.n_chirp * self.t_pri)

    @property
    def angle_resolution_deg(self) -> float:
        """가상 배열 기반 각도 분해능 (도)"""
        n_v = self.n_virtual
        if n_v < 2:
            return 180.0
        # λ/2 간격 n_v 소자 → Rayleigh criterion
        return np.degrees(2.0 / n_v)  # 대략적 근사

    def range_axis(self) -> np.ndarray:
        """Range-FFT bin → 거리 (m) 변환 축"""
        freqs = np.fft.fftfreq(self.n_range_fft, d=1.0 / self.fs_hz)
        return freqs[:self.n_range_fft // 2] * 3e8 / (2.0 * self.mu)

    def velocity_axis(self) -> np.ndarray:
        """Doppler-FFT bin → 속도 (m/s) 변환 축"""
        f_d = np.fft.fftshift(np.fft.fftfreq(self.n_doppler_fft, d=self.t_pri))
        return -f_d * self.lam / 2.0   # sign: 접근 = 양수

    def angle_axis(self) -> np.ndarray:
        """AoA-FFT bin → 각도 (도) 변환 축"""
        f_s = np.fft.fftshift(np.fft.fftfreq(self.n_angle_fft))
        sin_t = f_s * self.lam / self.d_elem
        sin_t = np.clip(sin_t, -1.0, 1.0)
        return np.degrees(np.arcsin(sin_t))

    def summary(self) -> str:
        return (
            f"fc={self.fc_hz/1e9:.1f}GHz  B={self.bandwidth_hz/1e6:.0f}MHz  "
            f"Tc={self.chirp_dur_s*1e6:.0f}μs  PRF={self.prf_hz:.0f}Hz\n"
            f"ΔR={self.range_resolution:.2f}m  Rmax={self.range_max_m:.1f}m  "
            f"Vmax=±{self.velocity_max_mps:.1f}m/s  ΔV={self.velocity_resolution:.2f}m/s"
        )
