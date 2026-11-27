import h5py,numpy as np
f=h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def load(g,key):
    return np.fromstring(g[key][()][0].decode().replace('[','').replace(']','').replace('\n',' '),sep=' ')
g=f['Q1']
Ig=load(g,'I_g'); Qg=load(g,'Q_g'); Ie=load(g,'I_e'); Qe=load(g,'Q_e')
ang0=g['Angle'][()][0]
best=None
for a in np.linspace(-np.pi,np.pi,721):
    for sign in [+1,-1]:
        rg=Ig*np.cos(a)+sign*Qg*np.sin(a)
        re=Ie*np.cos(a)+sign*Qe*np.sin(a)
        mg,me=np.median(rg),np.median(re)
        gclean=rg[rg<=mg] if mg<me else rg[rg>=mg]
        eclean=re[re>=me] if mg<me else re[re<=me]
        sg=np.std(gclean,ddof=1); se=np.std(eclean,ddof=1)
        sbar=np.sqrt((sg*sg+se*se)/2)
        snr=abs(me-mg)/sbar
        if best is None or snr>best[0]:
            best=(snr,a,sign,abs(me-mg),sbar,sg,se)
print('best SNR scan:', best)
print('stored angle:', ang0)
# also try rotation matrix form Icos-Qsin with angle
for a in [ang0]:
    for expr in ['Icos+Qsin','Icos-Qsin','Qcos+Isin','Qcos-Isin']:
        pass
# compute fidelity at stored angle both signs
for sign in [+1,-1]:
    rg=Ig*np.cos(ang0)+sign*Qg*np.sin(ang0)
    re=Ie*np.cos(ang0)+sign*Qe*np.sin(ang0)
    mg,me=np.median(rg),np.median(re)
    vals=np.sort(np.concatenate([rg,re]))
    bestacc=0
    for t in vals:
        if mg<me: acc=(np.sum(rg<=t)+np.sum(re>t))/(len(rg)+len(re))
        else: acc=(np.sum(rg>=t)+np.sum(re<t))/(len(rg)+len(re))
        if acc>bestacc: bestacc=acc
    print('sign',sign,'fidelity',bestacc)
