import h5py
import numpy as np

f = h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick_runs/agent_nemotron-35-super-vl/ss_ge.h5', 'r')
for k in sorted(f.keys()):
    g = f[k]
    print(f'=== {k} ===')
    print('  Fidelity:', g['Fidelity'][()])
    print('  I_e shape:', g['I_e'].shape, 'dtype:', g['I_e'].dtype)
    print('  I_g shape:', g['I_g'].shape, 'dtype:', g['I_g'].dtype)
    print('  Q_e shape:', g['Q_e'].shape, 'dtype:', g['Q_e'].dtype)
    print('  Q_g shape:', g['Q_g'].shape, 'dtype:', g['Q_g'].dtype)
    for attr in ['Angle', 'Batch Num', 'Round Num']:
        if attr in g:
            print(f'  {attr}:', g[attr][()])
    print()
f.close()
