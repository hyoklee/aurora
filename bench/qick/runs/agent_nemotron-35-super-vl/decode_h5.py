import h5py
import numpy as np
import pickle

f = h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick_runs/agent_nemotron-35-super-vl/ss_ge.h5', 'r')

# Try to decode I_e for Q1
g = f['Q1']
ie_raw = g['I_e'][()]
print('Type of I_e raw:', type(ie_raw))
print('I_e raw (first 200 bytes):', ie_raw[:200] if isinstance(ie_raw, bytes) else ie_raw)

# Try pickle
try:
    obj = pickle.loads(ie_raw)
    print('Pickle loaded type:', type(obj))
    if isinstance(obj, np.ndarray):
        print('  shape:', obj.shape, 'dtype:', obj.dtype)
        print('  first 5:', obj[:5])
    elif isinstance(obj, dict):
        print('  keys:', obj.keys())
    elif isinstance(obj, list):
        print('  len:', len(obj), 'first type:', type(obj[0]) if obj else 'N/A')
except Exception as e:
    print('Pickle failed:', e)

# Try numpy frombuffer
try:
    arr = np.frombuffer(ie_raw, dtype=np.float64)
    print('frombuffer float64: shape:', arr.shape, 'first 5:', arr[:5])
except Exception as e:
    print('frombuffer float64 failed:', e)

f.close()
