import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = '/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick_runs/agent_nemotron-35-super-vl/ss_ge.h5'
OUT_PATH = '/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick_runs/agent_nemotron-35-super-vl/readout.png'

def get_scalar(ds):
    """Extract a scalar from an HDF5 dataset that may be a 1-element array."""
    val = ds[()]
    if isinstance(val, np.ndarray):
        return val.item()
    return val

def decode_byte_array(raw):
    """Decode a byte string of newline-separated floats into a numpy array."""
    if isinstance(raw, bytes):
        text = raw.decode('utf-8')
    elif isinstance(raw, np.ndarray) and raw.dtype.kind == 'S':
        if raw.size == 1:
            text = raw.item().decode('utf-8')
        else:
            text = b'\n'.join(raw).decode('utf-8')
    else:
        text = str(raw)
    text = text.replace('[', ' ').replace(']', ' ')
    values = [float(x) for x in text.split() if x.strip()]
    return np.array(values)

def compute_fidelity(I_g, Q_g, I_e, Q_e, angle):
    """Compute single-shot readout fidelity from raw I/Q shots."""
    Z_g = I_g + 1j * Q_g
    Z_e = I_e + 1j * Q_e
    
    Z_g_rot = Z_g * np.exp(-1j * angle)
    Z_e_rot = Z_e * np.exp(-1j * angle)
    
    x_g = np.real(Z_g_rot)
    x_e = np.real(Z_e_rot)
    
    all_vals = np.sort(np.concatenate([x_g, x_e]))
    
    best_fid = 0.0
    best_thresh = 0.0
    
    for i in range(len(all_vals) - 1):
        thresh = (all_vals[i] + all_vals[i+1]) / 2
        for swap in [False, True]:
            if swap:
                pred_g = x_g < thresh
                pred_e = x_e > thresh
            else:
                pred_g = x_g > thresh
                pred_e = x_e < thresh
            
            p_correct_g = np.mean(pred_g)
            p_correct_e = np.mean(pred_e)
            fid = (p_correct_g + p_correct_e) / 2
            
            if fid > best_fid:
                best_fid = fid
                best_thresh = thresh
    
    return best_fid, best_thresh, x_g, x_e

f = h5py.File(DATA_PATH, 'r')

results = {}
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Single-Shot Qubit Readout - Recomputed Fidelity', fontsize=16)

for idx, k in enumerate(sorted(f.keys())):
    g = f[k]
    
    I_g = decode_byte_array(g['I_g'][()])
    Q_g = decode_byte_array(g['Q_g'][()])
    I_e = decode_byte_array(g['I_e'][()])
    Q_e = decode_byte_array(g['Q_e'][()])
    angle = get_scalar(g['Angle'])
    stored_fid = get_scalar(g['Fidelity'])
    
    print(f"=== {k} ===")
    print(f"  N_g = {len(I_g)}, N_e = {len(I_e)}")
    print(f"  Stored fidelity: {stored_fid:.4f}")
    print(f"  Angle: {angle:.4f}")
    
    fid, thresh, x_g, x_e = compute_fidelity(I_g, Q_g, I_e, Q_e, angle)
    print(f"  Recomputed fidelity: {fid:.4f} (threshold={thresh:.2f})")
    print(f"  Match stored: {abs(fid - stored_fid) < 1e-4}")
    
    results[k] = {
        'stored_fidelity': stored_fid,
        'recomputed_fidelity': fid,
        'threshold': thresh,
        'N_g': len(I_g),
        'N_e': len(I_e),
        'angle': angle,
    }
    
    ax = axes[idx // 3, idx % 3]
    ax.hist(x_g, bins=50, alpha=0.5, label='Ground', color='blue', density=True)
    ax.hist(x_e, bins=50, alpha=0.5, label='Excited', color='red', density=True)
    ax.axvline(thresh, color='black', linestyle='--', label=f'Threshold={thresh:.1f}')
    ax.set_xlabel('Rotated I (projection)')
    ax.set_ylabel('Density')
    ax.set_title(f'{k}: stored={stored_fid:.3f}, recomputed={fid:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"\nPlot saved to {OUT_PATH}")

print("\n=== SUMMARY ===")
print(f"{'Qubit':<8} {'Stored':<10} {'Recomputed':<12} {'Match':<8} {'N_g':<8} {'N_e':<8}")
for k in sorted(results.keys()):
    r = results[k]
    match = 'YES' if abs(r['recomputed_fidelity'] - r['stored_fidelity']) < 1e-4 else 'NO'
    print(f"{k:<8} {r['stored_fidelity']:<10.4f} {r['recomputed_fidelity']:<12.4f} {match:<8} {r['N_g']:<8} {r['N_e']:<8}")

f.close()
