#!/usr/bin/env python3
"""Entry point. All outputs are separate from previous AAF/CAENL/PBRC runs."""
import os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from aaf_strengthen.cli import main
if __name__=='__main__':main()
