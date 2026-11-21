import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
for q in f.keys():
    g=f[q]
    sc=g['Syst Config'][0].decode('utf-8','replace')
    # evaluate dict safely-ish
    d=eval(sc)
    print('\n',q)
    for key in ['list_of_all_qubits','qubit_ch','res_ch','ro_ch','res_gain_ge','res_length','qubit_gain_ge','qubit_pi_len','qubit_length_ge','steps','py_avg','relax_delay','threshold','res_freq_ge']:
        print('  ',key,'=',d.get(key))
    # exp config find 'ss_ge' or 'readout' maybe
    ec=g['Exp Config'][0].decode('utf-8','replace')
    print('  exp config len', len(ec))
