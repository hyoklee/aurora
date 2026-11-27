import h5py,numpy as np
f=h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def load(g,key):
    return np.fromstring(g[key][()][0].decode().replace('[','').replace(']','').replace('\n',' '),sep=' ')
for q in ['Q1','Q2','Q5']:
    g=f[q]
    Ig=load(g,'I_g'); Qg=load(g,'Q_g'); Ie=load(g,'I_e'); Qe=load(g,'Q_e')
    ang=g['Angle'][()][0]
    dI=Ie.mean()-Ig.mean(); dQ=Qe.mean()-Qg.mean()
    cent_angle=np.arctan2(dQ,dI)
    print(q,'stored angle',ang,'centroid angle',cent_angle,'diff',ang-cent_angle)
    # angle might be rotation to make g/e horizontal: rotate by -centroid_angle?
    for a in [ang, cent_angle, -ang, -cent_angle, ang+np.pi/2]:
        rg=Ig*np.cos(a)+Qg*np.sin(a); re=Ie*np.cos(a)+Qe*np.sin(a)
        mg,me=np.median(rg),np.median(re)
        print(f'   a={a:.4f}: dmu={abs(me-mg):.2f}')
