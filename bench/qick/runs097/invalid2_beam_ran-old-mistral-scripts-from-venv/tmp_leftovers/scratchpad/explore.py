import h5py, numpy as np

def load(q,name):
    f=h5py.File('ss_ge.h5','r')
    raw=f[q][name][()]
    s=raw[0].decode() if isinstance(raw,np.ndarray) else raw.decode()
    s=s.replace('[','').replace(']','')
    return np.fromstring(s, sep=' ')

f=h5py.File('ss_ge.h5','r')
q='Q1'
angle=float(f[q]['Angle'][0])
Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
print('angle',angle)
for label, rg, re_ in [
    ('+ (I cos + Q sin)', Ig*np.cos(angle)+Qg*np.sin(angle), Ie*np.cos(angle)+Qe*np.sin(angle)),
    ('- (I cos - Q sin)', Ig*np.cos(angle)-Qg*np.sin(angle), Ie*np.cos(angle)-Qe*np.sin(angle)),
    ('I cosθ + Q sinθ? same', None, None),
]:
    if rg is None: continue
    print('\n',label)
    print('  mean_g,mean_e,sep', rg.mean(), re_.mean(), abs(re_.mean()-rg.mean()))
    print('  std_g,std_e', rg.std(ddof=1), re_.std(ddof=1))
    # clean side: g I<=median_g, e I>=median_e
    mg=np.median(rg); me=np.median(re_)
    cg=rg[rg<=mg]; ce=re_[re_>=me]
    print('  medians', mg, me, 'nclean', cg.size, ce.size)
    print('  clean std_g,std_e', cg.std(ddof=1), ce.std(ddof=1))
    print('  SNR', abs(re_.mean()-rg.mean())/np.sqrt((cg.std(ddof=1)**2+ce.std(ddof=1)**2)/2))
