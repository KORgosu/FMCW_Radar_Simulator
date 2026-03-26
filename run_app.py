"""
FMCW Radar Simulator 실행 스크립트
실행: python run_app.py (저장소 루트에서)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('QtAgg')

from app.main import main

if __name__ == '__main__':
    main()
