import h5py, numpy as np

def load(q,name):
    f=h5py.File('ss_ge.h5','r')
    raw=f[q][name][()]
    s=raw[0].decode() if isinstance(raw,np.ndarray) else raw.decode()
    s=s.replace('[','').replace(']','')
    return np.fromstring(s, sep=' ')

f=h5py.File('ss_ge.h5','r')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    angle=float(f[q]['Angle'][0])
    Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
    # standard rotation by angle: Irot = I cos - Q sin ; Qrot = I sin + Q cos
    Irot_g=Ig*np.cos(angle)-Qg*np.sin(angle)
    Irot_e=Ie*np.cos(angle)-Qe*np.sin(angle)
    Qrot_g=Ig*np.sin(angle)+Qg*np.cos(angle)
    Qrot_e=Ie*np.sin(angle)+Qe*np.cos(angle)
    print(f'{q} angle={angle:.4f}')
    print(f'   Irot sep_mean={abs(Irot_e.mean()-Irot_g.mean()):8.2f}  Qrot sep_mean={abs(Qrot_e.mean()-Qrot_g.mean()):8.2f}')
