import time, re, os
import numpy as np
import pandas as pd
import cte_golf as cg

WD='/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/golf_runs/agent_beam-1.0-fp8'
t0=time.time()
m=cg.manifest()
print('manifest', len(m), 'time', round(time.time()-t0,2))

def group_of(p):
    n=int(p[1:
