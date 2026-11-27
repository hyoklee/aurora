import h5py
f = h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
g=f['Q1']
for k in ['Exp Config','Syst Config']:
    s=g[k][()][0]
    if isinstance(s,bytes): s=s.decode()
    print('====',k)
    print(s)
