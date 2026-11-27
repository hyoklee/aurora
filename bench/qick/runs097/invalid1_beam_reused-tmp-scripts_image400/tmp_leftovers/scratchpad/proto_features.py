import time, re, os, sys
import numpy as np
import pandas as pd
import cte_golf as cg

ROOT='/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/golf_runs/agent_beam-1.0-fp8'
OUT='/tmp/scratchpad/features.parquet'
sys.path.insert(0, ROOT)

def group_of(pid):
    n=int(pid[1:
