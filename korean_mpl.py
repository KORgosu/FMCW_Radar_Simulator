"""Matplotlib에서 한글 축·제목이 깨지지 않도록 글꼴을 설정한다."""
from __future__ import annotations

import matplotlib
import matplotlib.font_manager as fm

_PREFERRED_FONTS = (
    "Malgun Gothic",
    "맑은 고딕",
    "NanumGothic",
    "Nanum Gothic",
    "Apple SD Gothic Neo",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Noto Sans CJK JP",
)


def configure_korean_font() -> None:
    """설치된 TTF 목록에서 우선순위대로 한글 글꼴 하나를 쓴다."""
    installed = {entry.name for entry in fm.fontManager.ttflist}
    for name in _PREFERRED_FONTS:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            break
    else:
        sans = matplotlib.rcParams.get("font.sans-serif", [])
        if isinstance(sans, str):
            sans = [sans]
        priority = (
            "Malgun Gothic",
            "Apple SD Gothic Neo",
            "Nanum Gothic",
            "DejaVu Sans",
        )
        merged = list(priority) + [s for s in sans if s not in priority]
        matplotlib.rcParams["font.sans-serif"] = merged
        matplotlib.rcParams["font.family"] = "sans-serif"

    matplotlib.rcParams["axes.unicode_minus"] = False
