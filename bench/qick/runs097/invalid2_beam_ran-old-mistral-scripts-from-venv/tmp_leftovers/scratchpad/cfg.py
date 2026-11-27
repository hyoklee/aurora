import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    raw=f[q]['Syst Config'][0]
    s=raw.decode() if isinstance(raw,np.ndarray) else raw.decode() if hasattr(raw,'decode') else str(raw)
    # eval with numpy in scope
    d=eval(s, {"np":np, "__builtins__":__builtins__})
    print('===',q)
    for k in ['qubit_ch','list_of_all_qubits','ro_ch','res_gain_ge','res_freq_ge','threshold','qubit_freq_ge','T1' if 'T1' in d else 'x']:
        if k in d: print('  ',k,'=',d[k])
