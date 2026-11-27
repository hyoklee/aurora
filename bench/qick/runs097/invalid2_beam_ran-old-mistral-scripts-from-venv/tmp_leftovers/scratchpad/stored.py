import h5py, numpy as np, ast

def load(q,name):
    f=h5py.File('ss_ge.h5','r')
    raw=f[q][name][()]
    s=raw[0].decode() if isinstance(raw,np.ndarray) else raw.decode()
    s=s.replace('[','').replace(']','')
    return np.fromstring(s, sep=' ')

f=h5py.File('ss_ge.h5','r')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    qi=int(q[1])-1
    g=f[q]
    angle=float(g['Angle'][0])
    Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
    sc=eval(g['Syst Config'][0].decode(),{"np":np,"__builtins__":__builtins__})
    thr_list=sc['threshold']
    thr_q=thr_list[qi] if qi < len(thr_list) else thr_list[0]
    stored=float(g['Fidelity'][0])
    # rotated sign + and -
    for sign in [+1,-1]:
        rg=Ig*np.cos(angle)+sign*Qg*np.sin(angle)
        re_=Ie*np.cos(angle)+sign*Qe*np.sin(angle)
        # threshold thr_q
        acc=(np.sum(rg<=thr_q)+np.sum(re_>thr_q))/(len(rg)+len(re_))
        print(f'{q} sign={sign:+d} thr_q={thr_q:6.2f} rotated acc={acc:.4f} (stored {stored:.4f})')
    # raw I threshold thr_q
    acc=(np.sum(Ig<=thr_q)+np.sum(Ie>thr_q))/(len(Ig)+len(Ie))
    print(f'   rawI thr_q={thr_q:6.2f} acc={acc:.4f}')
    # raw I threshold 0
    acc=(np.sum(Ig<=0)+np.sum(Ie>0))/(len(Ig)+len(Ie))
    print(f'   rawI thr0 acc={acc:.4f}')
