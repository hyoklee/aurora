import h5py, numpy as np, io
f=h5py.File('ss_ge.h5','r')
def parse(s):
    s=s.strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')
def load(q,k):
    s=f[q][k][()][0]
    if isinstance(s,(bytes,bytearray)): s=s.decode()
    return parse(s)
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
    ang=float(np.asarray(f[q]['Angle'][()]).ravel()[0])
    # separation vector in raw
    dg=np.array([Ie.mean()-Ig.mean(), Qe.mean()-Qg.mean()])
    mag=np.linalg.norm(dg)
    print(f'{q}: angle={ang:.4f} raw_sep_mag={mag:.1f} storedF={float(np.asarray(f[q]["Fidelity"][()]).ravel()[0]):.4f}')
    # try rotation: new I = I*cos? Let's compute both conventions separation along x after rotating by -angle or +angle
    for sign in [+1,-1]:
        c=np.cos(sign*ang); s=np.sin(sign*ang)
        # rotate points: x = I*c - Q*s? or I*c + Q*s
        for conv in ['IpQs','ImQs']:
            if conv=='IpQs':
                Igx=Ig*c+Qg*s; Iex=Ie*c+Qe*s
            else:
                Igx=Ig*c-Qg*s; Iex=Ie*c-Qe*s
            sep=Iex.mean()-Igx.mean()
            # clean-side: for g: I<=median(g); for e: I>=median(e) on rotated x? use original? We'll just print sep and std
            print(f'   sign={sign} conv={conv}: sep_x={sep:.1f}')
