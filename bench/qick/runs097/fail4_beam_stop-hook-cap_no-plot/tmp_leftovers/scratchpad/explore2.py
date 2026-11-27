import h5py,numpy as np
f=h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def load(g,key):
    return np.fromstring(g[key][()][0].decode().replace('[','').replace(']','').replace('\n',' '),sep=' ')
for q in ['Q1']:
    g=f[q]
    Ig=load(g,'I_g'); Qg=load(g,'Q_g'); Ie=load(g,'I_e'); Qe=load(g,'Q_e')
    ang=g['Angle'][()][0]
    for sgn in [+1,-1]:
        rg=Ig*np.cos(ang)+sgn*Qg*np.sin(ang)  # + for plus? test both
        # Actually two candidate rotations:
        pass
    for label,rotg,rote in [('Icos+Qsin', Ig*np.cos(ang)+Qg*np.sin(ang), Ie*np.cos(ang)+Qe*np.sin(ang)),
                            ('Icos-Qsin', Ig*np.cos(ang)-Qg*np.sin(ang), Ie*np.cos(ang)-Qe*np.sin(ang))]:
        mg=np.median(rotg); me=np.median(rote)
        # clean sides
        g_clean = rotg[rotg<=mg]
        e_clean = rote[rote>=me]
        sg=np.std(g_clean,ddof=1); se=np.std(e_clean,ddof=1)
        sbar=np.sqrt((sg*sg+se*se)/2)
        dmu=abs(me-mg)
        snr=dmu/sbar
        print(q,label,'mg',mg,'me',me,'dmu',dmu,'sg',sg,'se',se,'sbar',sbar,'SNR',snr,'nclean_g',len(g_clean),'nclean_e',len(e_clean))
