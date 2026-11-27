import h5py,numpy as np
f=h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def load(g,key):
    return np.fromstring(g[key][()][0].decode().replace('[','').replace(']','').replace('\n',' '),sep=' ')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g=f[q]
    Ig=load(g,'I_g'); Qg=load(g,'Q_g'); Ie=load(g,'I_e'); Qe=load(g,'Q_e')
    ang=g['Angle'][()][0]
    print('\n==',q,'angle',ang,'stored F',g['Fidelity'][()][0])
    for name,rot in [('Icos+Qsin', lambda I,Q: I*np.cos(ang)+Q*np.sin(ang)),
                     ('Icos-Qsin', lambda I,Q: I*np.cos(ang)-Q*np.sin(ang))]:
        rg=rot(Ig,Qg); re=rot(Ie,Qe)
        mg,me=np.median(rg),np.median(re)
        vals=np.sort(np.concatenate([rg,re]))
        best=0; bestt=None
        for t in vals:
            if mg<me:
                acc=(np.sum(rg<=t)+np.sum(re>t))/(len(rg)+len(re))
            else:
                acc=(np.sum(rg>=t)+np.sum(re<t))/(len(rg)+len(re))
            if acc>best:
                best=acc; bestt=t
        print(f'  rot={name}: mg={mg:.2f} me={me:.2f} dmu={abs(me-mg):.2f} best_acc={best:.4f} thr={bestt:.2f}')
